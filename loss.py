import torch
from torch import nn


def bce_loss_disc(pred_real, pred_fake):
    
    criterion = nn.BCEWithLogitsLoss()
    label_real = torch.ones_like(pred_real)
    label_fake = torch.zeros_like(pred_fake)
    
    loss_real = criterion(pred_real, label_real)
    loss_fake = criterion(pred_fake, label_fake)
    
    return (loss_real + loss_fake) / 2.


def bce_loss_gen(pred_fake):
    
    criterion = nn.BCEWithLogitsLoss()
    label_real = torch.ones_like(pred_fake)
    
    loss = criterion(pred_fake, label_real)
    
    return loss


def wasserstein_loss_disc(pred_real, pred_fake):
    
    loss = torch.mean(pred_fake) - torch.mean(pred_real)
    
    return loss


def wasserstein_loss_gen(pred_fake):
    
    loss = -torch.mean(pred_fake)
    
    return loss


def gradient_penalty_disc(x_real, x_fake, disc, epsilon, lambda_gp=10):
    
    x_hat = epsilon * x_real + (1 - epsilon) * x_fake
    pred_x_hat = disc(x_hat)
    grad = torch.autograd.grad(
        outputs=pred_x_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(pred_x_hat),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    grad = grad.view(len(grad), -1)
    grad_norm = grad.norm(2, dim=1)
    
    reg = torch.mean((grad_norm - 1) ** 2)
    
    return lambda_gp * reg