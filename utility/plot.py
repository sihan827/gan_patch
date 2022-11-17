import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from .normalize import normalize_tensor


def plot_loss_curve(fig, mean1, std1, label1, mean2, std2, label2, path=None):
    """ Plot two loss curves with mean and std """
    
    plt.figure(fig.number)
    
    x_len = len(mean1)
    
    min1 = mean1 - std1
    max1 = mean1 + std1
    
    min2 = mean2 - std2
    max2 = mean2 + std2
    
    plt.plot(range(x_len), mean1, label=label1, color='red')
    plt.fill_between(range(x_len), min1, max1, alpha=0.2, color='red')
    
    plt.plot(range(x_len), mean2, label=label2, color='blue')
    plt.fill_between(range(x_len), min2, max2, alpha=0.2, color='blue')
    
    plt.legend()
    
    if path is not None:
        
        fig.savefig(path)
        
    
def plot_image_grid(fig, batch_tensor, path=None):
    """ Normalize, reshape and plot batch image tensor """
    
    plt.figure(fig.number)
    
    num_image = batch_tensor.shape[0]
    num_row = int(np.ceil(np.sqrt(num_image)))
    
    image_tensor = normalize_tensor(batch_tensor, min=0, max=1)
    image_tensor = batch_tensor.detach().cpu()
    grid_tensor = make_grid(image_tensor, nrow=num_row)
    grid_tensor = grid_tensor.permute(1, 2, 0)
    
    plt.imshow(grid_tensor)
    
    if path is not None:
        
        fig.savefig(path)
    
    
def plot_fid_curve(fig, fid_score, label, path=None):
    """ Plot FID score """
    
    plt.figure(fig.number)
    
    x_len = len(fid_score)   
    plt.plot(range(x_len), fid_score, label=label, color='red')
    plt.legend()
    
    if path is not None:
        
        fig.savefig(path)
    
    