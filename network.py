import torch
import torch.nn.functional as F
from torch import nn
from math import log2

def set_network(model, z_dim=128, img_size=32, img_channel=1, num_feature=16, device='cuda'):
    """ set network models based on model parameter """
    
    if 'Generator' in model:
        if 'GeneratorDCGAN' in model:
            return GeneratorDCGAN(z_dim, img_size, img_channel, num_feature)
        
        elif 'PatchGenerator' in model:
            patch_size = int(model.split('.')[-1])
            coord_method = model.split('.')[-2]
            
            return PatchGenerator(z_dim, img_size, img_channel, num_feature, patch_size, coord_method, device)
        
        else:
            raise NotImplementedError
    
    if 'Discriminator' in model:
        if 'DiscriminatorDCGAN' in model:
            if 'BatchNorm' in model:
                return DiscriminatorDCGAN(img_size, img_channel, num_feature)
            
            else:
                return DiscriminatorDCGAN(img_size, img_channel, num_feature, use_batchnorm=False)
        
        elif 'PatchDiscriminator' in model:
            if model.split('.')[-2] == 'none':
                coord_method = 'none'
                classifier_method = 'conv'
                patch_grid_size = int(model.split('.')[-1])
                
            else:
                coord_method = model.split('.')[-3]
                classifier_method = model.split('.')[-2]
                patch_grid_size = int(model.split('.')[-1])
                
            use_batchnorm = True if 'BatchNorm' in model else False                  
                    
            return PatchDiscriminator(img_size, img_channel, num_feature, patch_grid_size, coord_method, classifier_method, use_batchnorm, device)
                
        else:
            raise NotImplementedError


class GeneratorDCGAN(nn.Module):

    def __init__(self, z_dim=128, img_size=32, img_channel=1, num_feature=16):

        super(GeneratorDCGAN, self).__init__()
        
        assert img_size >= 32, "Image size must be bigger than 32x32"
        self.z_dim = z_dim
        
        layers = []
        num_hidden = int(log2(img_size)) - 3
        layers.append(self.make_block(z_dim, 2 ** num_hidden * num_feature, 4, 1, 0))
        
        for n in reversed(range(num_hidden)):
            layers.append(self.make_block(2 ** (n + 1) * num_feature, 2 ** n * num_feature, 4, 2, 1))
        
        layers.append(self.make_block(num_feature, img_channel, 4, 2, 1, final_layer=True))
        
        self.network = nn.Sequential(*layers)
        
        self.weight_init()

    def make_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Tanh()
            )
    
    def weight_init(self):
        
        for layer in self.network:           
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal_(layer.weight.data, 0., 0.02)
                
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, 1., 0.02)
                nn.init.constant_(layer.bias.data, 0.)

    def forward(self, x):

        x = x.view(len(x), self.z_dim, 1, 1)
        y = self.network(x)

        return y
    

class DiscriminatorDCGAN(nn.Module):
    
    def __init__(self, img_size=32, img_channel=1, num_feature=16, use_batchnorm=True):

        super(DiscriminatorDCGAN, self).__init__()
        
        assert img_size >= 32, "Image size must be bigger than 32x32"
        
        layers = []
        num_hidden = int(log2(img_size)) - 3
        layers.append(nn.Sequential(
            nn.Conv2d(img_channel, num_feature, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        for n in range(num_hidden):
            layers.append(self.make_block(2 ** n * num_feature, 2 ** (n + 1) * num_feature, 4, 2, 1, use_batchNorm=use_batchnorm))
        
        layers.append(self.make_block(2 ** num_hidden * num_feature, 1, 4, 1, 0, final_layer=True))
        
        self.network = nn.Sequential(*layers)
        
        self.weight_init()

    def make_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, use_batchNorm=True, final_layer=False):

        if not final_layer:
            if use_batchNorm:          
                return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(output_channels),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )

            else:        
                return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
                
        else:

            return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def weight_init(self):
        
        for layer in self.network:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, 0., 0.02)
                
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, 1., 0.02)
                nn.init.constant_(layer.bias.data, 0.)
                
    def forward(self, x):

        y = self.network(x)

        return y
    

class PatchGenerator(nn.Module):
    
    def __init__(self, z_dim, img_size, img_channel=1, num_feature=16, patch_size=16, coord_method='one_hot', device='cuda'):
        
        super(PatchGenerator, self).__init__()
        
        assert img_size >= 32, "Image size must be bigger than 32x32"
        assert patch_size <= 32, "Patch size must be smaller than 32x32"
        
        self.z_dim = z_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid_size = img_size // patch_size
        self.num_patches = self.patch_grid_size ** 2
        self.coord_method = coord_method
        self.device = device
        
        if self.coord_method == 'one_hot':
            dim_add = self.num_patches
        
        elif self.coord_method == 'coord_value':
            dim_add = 2
        
        else:
            raise NotImplementedError
        
        layers = []
        num_del = 5 - int(log2(patch_size)) + 3
        num_hidden = int(log2(img_size)) - num_del
        layers.append(self.make_block(z_dim + dim_add, 2 ** num_hidden * num_feature, 4, 1, 0))
        
        for n in reversed(range(num_hidden)):
            layers.append(self.make_block(2 ** (n + 1) * num_feature, 2 ** n * num_feature, 4, 2, 1))
        
        layers.append(self.make_block(num_feature, img_channel, 4, 2, 1, final_layer=True))
        
        self.network = nn.Sequential(*layers)
        
        self.weight_init()
    
    def make_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
    
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Tanh()
            )
    
    def weight_init(self):
        
        for layer in self.network:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, 0., 0.02)
                
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, 1., 0.02)
                nn.init.constant_(layer.bias.data, 0.)              
    
    def add_coord_info_to_z(self, z, batch_size):
        
        if self.coord_method == 'one_hot':
            classes = F.one_hot(torch.arange(0, self.num_patches)).repeat(batch_size, 1).to(self.device)
            z_with_coords = torch.cat([z, classes], dim=1)
        
        elif self.coord_method == 'coord_value':
            x, y = torch.meshgrid(torch.arange(0, self.patch_grid_size), torch.arange(0, self.patch_grid_size))
            x = (x.unsqueeze(dim=0) / (self.patch_grid_size - 1)).to(self.device)
            y = (y.unsqueeze(dim=0) / (self.patch_grid_size - 1)).to(self.device)
            coords = torch.cat([x, y]).transpose(0, 2).reshape((-1, 2)).repeat(batch_size, 1)
            z_with_coords = torch.cat([z, coords], dim=1)
            
        else:
            raise NotImplementedError
        
        return z_with_coords
    
    def patch_forward(self, x):
        
        x = x.view(len(x), -1, 1, 1)
        y = self.network(x)

        return y

    def forward(self, x):
        
        batch_size = len(x)
        x = torch.cat([x] * self.num_patches, dim=1).reshape(batch_size * self.num_patches, -1)
        x = self.add_coord_info_to_z(x, batch_size)
        x = self.patch_forward(x)
        x = torch.cat(list(x), dim=2)
        x = x.chunk(batch_size * self.patch_grid_size, dim=2)
        x = torch.cat(x, dim=1)
        x = x.chunk(batch_size, dim=1)
        y = torch.stack(list(x), dim=0)
        
        return y
    

class PatchDiscriminator(nn.Module):
    
    def __init__(self, img_size=32, img_channel=1, num_feature=16, patch_grid_size=2, coord_method='one_hot', classifier_method='linear', use_batchnorm=True, device='cuda'):
        
        super(PatchDiscriminator, self).__init__()
        
        assert img_size >= 32, "Image size must be bigger than 32x32"
        assert patch_grid_size <= 32, "Patch grid size must be smaller than 32x32"
        
        self.coord_method = coord_method
        self.classifier_method = classifier_method
        self.device = device
        
        layers = []
        num_del = int(log2(patch_grid_size)) + 1
        num_hidden = int(log2(img_size)) - num_del
        layers.append(nn.Sequential(
            nn.Conv2d(img_channel, num_feature, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        for n in range(num_hidden):
            layers.append(self.make_block(2 ** n * num_feature, 2 ** (n + 1) * num_feature, 4, 2, 1, use_batchnorm=use_batchnorm))
            num_out = 2 ** (n + 1) * num_feature
        
        self.convs = nn.Sequential(*layers)
        
        if self.coord_method == 'one_hot':
            if self.classifier_method == 'linear':
                self.classifier = nn.Sequential(
                    nn.Linear(num_out + patch_grid_size ** 2, 1)
                    # nn.Linear(num_out + 4, num_feature * 2),
                    # nn.ReLU(inplace=True),
                    # nn.Linear(num_feature * 2, 1)
                )
                
            elif self.classifier_method == 'conv':
                self.classifier = nn.Conv2d(num_out + patch_grid_size ** 2, 1, kernel_size=1, stride=1)
            
            else:
                raise NotImplementedError
        
        elif self.coord_method == 'coord_value':
            if self.classifier_method == 'linear':
                self.classifier = nn.Sequential(
                    nn.Linear(num_out + 2, 1)
                    # nn.Linear(num_out + 2, num_feature * 2),
                    # nn.ReLU(inplace=True),
                    # nn.Linear(num_feature * 2, 1)
                )
            
            elif self.classifier_method == 'conv':
                self.classifier = nn.Conv2d(num_out + 2, 1, kernel_size=1, stride=1)
            
            else:
                raise NotImplementedError
        
        elif self.coord_method == 'none':
            # If coordinate information is not added, it is similar to normal PatchGAN
            # So linear layer is useless
            self.classifier = nn.Conv2d(num_out, 1, kernel_size=1, stride=1)
        
        else:
            raise NotImplementedError

        self.weight_init()
            
    def make_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, use_batchnorm=True):
        
        if use_batchnorm:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
    
    def add_coord_info(self, feature_blocks):
        
        batch_size = feature_blocks.shape[0]
        coord_h = feature_blocks.shape[2]
        coord_w = feature_blocks.shape[3]
        
        if self.coord_method == 'one_hot':
            num_patches = coord_h * coord_w
            classes = F.one_hot(torch.arange(0, num_patches)).reshape(num_patches, coord_h, coord_w).to(self.device)
            classes = torch.unsqueeze(classes, dim=0).repeat(batch_size, 1, 1, 1)            
            feature_with_coords = torch.cat([feature_blocks, classes], dim=1)
            
            if self.classifier_method == 'linear':
                feature_with_coords = feature_with_coords.transpose(1, 3).reshape(-1, feature_with_coords.shape[1])
        
        elif self.coord_method == 'coord_value':
            x, y = torch.meshgrid(torch.arange(0, coord_h), torch.arange(0, coord_w))
            x = (x.unsqueeze(dim=0) / (coord_h - 1)).to(self.device)
            y = (y.unsqueeze(dim=0) / (coord_w - 1)).to(self.device)
            
            coord_grid = torch.cat([x, y]).unsqueeze(dim=0).repeat(batch_size, 1, 1, 1)
            feature_with_coords = torch.cat([feature_blocks, coord_grid], dim=1)
            
            if self.classifier_method == 'linear':
                feature_with_coords = feature_with_coords.transpose(1, 3).reshape(-1, feature_with_coords.shape[1])
        
        elif self.coord_method == 'none':
            feature_with_coords = feature_blocks
        
        return feature_with_coords
            
    def weight_init(self):
        
        for layer in self.convs:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, 0., 0.02)
                
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, 1., 0.02)
                nn.init.constant_(layer.bias.data, 0.)
    
    def forward(self, x):
        x = self.convs(x)
        x = self.add_coord_info(x)
        y = self.classifier(x)
        
        return y


