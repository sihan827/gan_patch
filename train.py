import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss import *
from utility.plot import *
from utility.fid import fid_score
import os.path


class Train:
    """ Basic training class of GAN """
    
    def __init__(self, models, params, loader, logger, option):
        
        # generator, discriminator, generator optimizer and discriminator optimizer
        self.disc = models['disc']
        self.gen = models['gen']
        self.optim_disc = models['optim_disc']
        self.optim_gen = models['optim_gen']
        
        # total iterations, dimension of latent vector, 
        # number of ex-sample, lambda of gradient penalty
        self.num_epoch = params['num_epoch']
        self.z_dim = params['z_dim']
        self.num_samples = params['num_samples']
        self.lambda_gp = params['lambda_gp']
        
        # dataloader for training
        self.loader = loader
        
        # logger for saving loss and FID score
        self.logger = logger
        
        # type of loss, usage of FID score, path of results, type of device
        self.loss_type = option['loss_type']
        self.use_fid = option['use_fid']
        self.result_path = option['result_path']
        self.device = option['device']
        
        # loss container for generator and discriminator
        self.gen_losses_mean = np.zeros(self.num_epoch)
        self.gen_losses_std = np.zeros(self.num_epoch)
        self.disc_losses_mean = np.zeros(self.num_epoch)
        self.disc_losses_std = np.zeros(self.num_epoch)
        
        # number of image for calculating FID, FID container
        if self.use_fid:
            self.num_fid = params['num_fid']
            self.fid_score = np.zeros(self.num_epoch)
        
    def train(self):
        
        self.gen.train()
        self.disc.train()
        
        z_fix = self.generate_latent(num_samples=self.num_samples)
        
        fig_img = plt.figure()
        fig_curve = plt.figure()
        fig_fid = plt.figure()
        
        for epoch in range(self.num_epoch):
            
            self.gen.train()

            disc_batch_losses = []
            gen_batch_losses = []
            
            for x_real, _ in tqdm(self.loader, ncols=50):            
                x_real = x_real.to(self.device)  
                batch_size = len(x_real)  
                
                # update discriminator
                self.optim_disc.zero_grad()
                       
                z = self.generate_latent(batch_size)
                x_fake = self.gen(z).detach()
                
                pred_real = self.disc(x_real)
                pred_fake = self.disc(x_fake)
                
                if self.loss_type == "cross_entropy":
                    disc_batch_loss = bce_loss_disc(pred_real, pred_fake)
                    
                elif self.loss_type == "wasserstein":
                    loss = wasserstein_loss_disc(pred_real, pred_fake)
                    eps = torch.randn(batch_size, 1, 1, 1, device=self.device, requires_grad=True)
                    reg = gradient_penalty_disc(x_real, x_fake, self.disc, eps, self.lambda_gp)
                    
                    disc_batch_loss = loss + reg
                
                else:
                    raise NotImplementedError
                
                disc_batch_loss.backward()
                self.optim_disc.step()
                
                # update generator
                self.optim_gen.zero_grad()
                
                z = self.generate_latent(batch_size)
                x_fake = self.gen(z)
                
                pred_fake = self.disc(x_fake)
                
                if self.loss_type == "cross_entropy":
                    gen_batch_loss = bce_loss_gen(pred_fake)
                    
                elif self.loss_type == "wasserstein":
                    gen_batch_loss = wasserstein_loss_gen(pred_fake)
                
                else:
                    raise NotImplementedError
                
                gen_batch_loss.backward()
                self.optim_gen.step()
                
                disc_batch_losses = np.append(disc_batch_losses, disc_batch_loss.item())
                gen_batch_losses = np.append(gen_batch_losses, gen_batch_loss.item())
            
            self.disc_losses_mean[epoch] = np.mean(disc_batch_losses)
            self.disc_losses_std[epoch] = np.std(disc_batch_losses)
            self.gen_losses_mean[epoch] = np.mean(gen_batch_losses)
            self.gen_losses_std[epoch] = np.std(gen_batch_losses)
            
            msg = '[%04d/%04d] (G) %5.3f, (D) %5.3f\n' % (
                epoch, self.num_epoch, self.gen_losses_mean[epoch], self.disc_losses_mean[epoch]
            )
            
            self.logger.write(msg)
            
            # generate sample images
            self.gen.eval()
            with torch.no_grad():
                image_fix = self.gen(z_fix)
            
            fig_img.clf()
            fig_curve.clf()
            
            image_path = os.path.join(self.result_path, 'image.png')
            loss_path = os.path.join(self.result_path, 'loss.png')
            
            plot_image_grid(fig_img, image_fix, path=image_path)
            plot_loss_curve(
                fig_curve, 
                self.gen_losses_mean, self.gen_losses_std, 'generator',
                self.disc_losses_mean, self.disc_losses_std, 'discriminator',
                path=loss_path
            )
            
            if self.use_fid:
                self.calculate_fid(epoch, fig_fid)
            
        plt.close(fig_img)
        plt.close(fig_curve)
        plt.close(fig_fid)
        
        self.logger.close()            
        
    def generate_latent(self, num_samples):
        """ Generate latent vectors """
        
        return torch.randn(num_samples, self.z_dim, device=self.device)

    def calculate_fid(self, epoch, figure):
        """ Calculate FID score using current generator """
        
        self.gen.eval()
        
        real_batch = []
        fake_batch = []
        
        for i, (real, _) in enumerate(self.loader):
            batch_size = len(real)
            z = self.generate_latent(batch_size)
            fake = self.gen(z)
            
            real_batch.append(real.type(torch.FloatTensor))
            fake_batch.append(fake.type(torch.FloatTensor))
            
            if i * batch_size > self.num_fid:
                break
        
        real_batch = torch.cat(real_batch, dim=0)
        fake_batch = torch.cat(fake_batch, dim=0)
            
        fid = fid_score.calculate_fid_given_batches(real_batch, fake_batch, batch_size=32)
        self.fid_score[epoch] = fid
        
        txt_fid_path = os.path.join(self.result_path, 'fid.txt')
        txt_fid = open(txt_fid_path, 'at')
        txt_fid.write('{}: {}\n'.format(epoch, fid))
        txt_fid.close()
        
        message = '[%04d/%04d] (FID) %5.3f\n' % (epoch, self.num_epoch, self.fid_score[epoch])
        self.logger.write(message)
        
        figure.clf()        
        fig_fid_path = os.path.join(self.result_path, 'fid.png')
        plot_fid_curve(figure, self.fid_score, 'fid', path=fig_fid_path)
        
        
            
        
        
        