import torch


def normalize_tensor(input_tensor, min=0, max=1):
    """ Normalize input tensor to range [0, 1] """
    
    batch_size = input_tensor.shape[0]
    c = input_tensor.shape[1]
    h = input_tensor.shape[2]
    w = input_tensor.shape[3]
    
    tensor_reshape = torch.reshape(input_tensor, (batch_size, c * h * w))
    
    tensor_min = torch.min(tensor_reshape, dim=1).values.unsqueeze(dim=1)
    tensor_normalize = tensor_reshape - tensor_min
    tensor_max = torch.max(tensor_normalize, dim=1).values.unsqueeze(dim=1)
    tensor_normalize = tensor_normalize / tensor_max
    
    a = torch.tensor(max) - torch.tensor(min)
    b = torch.tensor(min)
    
    tensor_normalize = a * tensor_normalize + b
    tensor_normalize = torch.reshape(tensor_normalize, (batch_size, c, h, w))
    
    return tensor_normalize


def whiten_tensor(input_tensor, mean=0, std=1):
    """ Normalize input tensor to mean = 0, std = 1 """
    
    batch_size = input_tensor.shape[0]
    c = input_tensor.shape[1]
    h = input_tensor.shape[2]
    w = input_tensor.shape[3]
    
    tensor_reshape = torch.reshape(input_tensor, (batch_size, c * h * w))
    
    tensor_mean = torch.mean(tensor_reshape, dim=1).unsqueeze(dim=1)
    tensor_std = torch.mean(tensor_reshape, dim=1).unsqueeze(dim=1)
    
    tensor_normalize = (tensor_reshape - tensor_mean) / tensor_std
    tensor_normalize = (tensor_normalize + mean) * std
    tensor_normalize = torch.reshape(tensor_normalize, (batch_size, c, h, w))
    
    return tensor_normalize
    
    