import torch
from torchvision import transforms
from torchvision.datasets import *
from torch.utils.data import DataLoader
import warnings

import configparser
import sys
import os
import datetime

from network import set_network
from train import Train

from utility.logger import Logger

warnings.filterwarnings('ignore')

# load option file
option_file = './option.ini'
if len(sys.argv) == 2:
    option_file = sys.argv[1]

config = configparser.ConfigParser()
config.read(option_file)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# parse option file
option_dataset = {
    'name' : config.get('Dataset', 'Dataset.name'),
    'directory' : config.get('Dataset', 'Dataset.directory'),
    'batch_size' : int(config.get('Dataset', 'Dataset.batch_size')),
    'dim_latent' : int(config.get('Dataset', 'Dataset.dim_latent')),
    'image_channel' : int(config.get('Dataset', 'Dataset.image_channel')),
    'image_size' : int(config.get('Dataset', 'Dataset.image_size'))
}

option_network = {
    'num_feature' : int(config.get('Network', 'Network.num_feature')),
    'generator' : config.get('Network', 'Network.generator'),
    'discriminator' : config.get('Network', 'Network.discriminator')
}

option_optimizer = {
    'algorithm' : config.get('Optimizer', 'Optimizer.algorithm'),
    'number_epoch' : int(config.get('Optimizer', 'Optimizer.number_epoch')),
    'weight_decay' : float(config.get('Optimizer', 'Optimizer.weight_decay')),
    'momentum' : float(config.get('Optimizer', 'Optimizer.momentum')),
    'beta1' : float(config.get('Optimizer', 'Optimizer.beta1')),
    'beta2' : float(config.get('Optimizer', 'Optimizer.beta2')),
    'generator_learning_rate' : float(config.get('Optimizer', 'Optimizer.generator_learning_rate')),
    'discriminator_learning_rate' : float(config.get('Optimizer', 'Optimizer.discriminator_learning_rate'))
}

option_loss = {
    'loss_fidelity' : config.get('Loss', 'Loss.loss_fidelity'),
    'lambda_gradient_penalty' : int(config.get('Loss', 'Loss.lambda_gradient_penalty'))
}

option_result = {
    'save' : bool(config.get('Result', 'Result.save')),
    'path' : config.get('Result', 'Result.path'),
    'use_fid' : bool(config.get('Result', 'Result.use_FID')),
    'nexamples_fid' : int(config.get('Result', 'Result.nexamples_FID'))
}

# make a result directory
date_and_time = datetime.datetime.now()
title = date_and_time.strftime('%Y-%m-%d_%H-%M-%S')

if not os.path.exists(option_result['path']):
    os.makedirs(option_result['path'])

result_path = os.path.join(option_result['path'], title)
os.makedirs(result_path)

# load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(option_dataset['image_size']),
])

if option_dataset['name'].lower() == 'MNIST'.lower():
    dataset = MNIST(option_dataset['directory'], download=False, transform=transform)

elif option_dataset['name'].lower() == 'CIFAR10'.lower():
    dataset = CIFAR10(option_dataset['directory'], download=False, transform=transform)

elif option_dataset['name'].lower() == 'FFHQ'.lower():
    dataset_path = os.path.join(option_dataset['directory'], 'ffhq_thumbs')
    dataset = ImageFolder(dataset_path, transform=transform)

else:
    raise NotImplementedError

# make a dataloader for dataset
dataloader = DataLoader(
    dataset,
    batch_size=option_dataset['batch_size'],
    num_workers=2
)

# set network models
generator = set_network(
    model=option_network['generator'],
    z_dim=option_dataset['dim_latent'], 
    img_size=option_dataset['image_size'],
    img_channel=option_dataset['image_channel'],
    num_feature=option_network['num_feature'],
    device=device
).to(device)

discriminator = set_network(
    model=option_network['discriminator'],
    img_size=option_dataset['image_size'],
    img_channel=option_dataset['image_channel'],
    num_feature=option_network['num_feature'],
    device=device
).to(device)

print('Generator     :', option_network['generator'])
print('Discriminator :', option_network['discriminator'])
print('Loss          :', option_loss['loss_fidelity'])

# set optiminzers
g_lr = option_optimizer['generator_learning_rate']
d_lr = option_optimizer['discriminator_learning_rate']

if option_optimizer['algorithm'].lower() == 'SGD'.lower():
    print('Optimizer     : SGD')
    momentum = option_optimizer['momentum']
    weight_decay = option_optimizer['weight_decay']
    
    optim_generator = torch.optim.SGD(
        generator.parameters(), lr=g_lr, weight_decay=weight_decay, momentum=momentum
    )
    optim_discriminator = torch.optim.SGD(
        discriminator.parameters(), lr=d_lr, weight_decay=weight_decay, momentum=momentum
    )

elif option_optimizer['algorithm'].lower() == 'Adam'.lower():
    print('Optimizer     : Adam')
    weight_decay = option_optimizer['weight_decay']
    beta1 = option_optimizer['beta1']
    beta2 = option_optimizer['beta2']
    
    optim_generator = torch.optim.Adam(
        generator.parameters(), lr=g_lr, betas=(beta1, beta2), weight_decay=weight_decay
    )
    optim_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=d_lr, betas=(beta1, beta2), weight_decay=weight_decay
    )

# set seed
torch.manual_seed(0)

# set parameters for Train class
logger_path = os.path.join(result_path, 'log.txt')
logger = Logger(logger_path)

models = {
    'disc' : discriminator,
    'gen' : generator,
    'optim_disc' : optim_discriminator,
    'optim_gen' : optim_generator
}

params = {
    'num_epoch' : option_optimizer['number_epoch'],
    'z_dim' : option_dataset['dim_latent'],
    'num_samples' : 36,
    'num_fid' : option_result['nexamples_fid'],
    'lambda_gp' : option_loss['lambda_gradient_penalty']
}

option = {
    'loss_type' : option_loss['loss_fidelity'],
    'use_fid' : option_result['use_fid'],
    'result_path' : result_path,
    'device' : device
}

# set Train class with parameters and run  
trainer = Train(models, params, dataloader, logger, option)

trainer.train()

if option_result['save']:
    generator_path = os.path.join(result_path, 'gen.pth')
    discriminator_path = os.path.join(result_path, 'disc.pth')
    option_path = os.path.join(result_path, 'option.txt')
    
    torch.save(generator, generator_path)
    torch.save(discriminator, discriminator_path)
    
    with open(option_path, 'w') as configfile:
        config.write(configfile)



