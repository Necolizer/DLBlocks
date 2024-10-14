import torch
import numpy as np

data_tensor = torch.rand((3, 64, 25, 2))

def add_noise(tensor, mean=0, std=1):
    noise = torch.randn_like(tensor) * std + mean
    return tensor + noise

data_tensor = add_noise(data_tensor, mean=0, std=0.01)

def add_mask(tensor, mask_prob=0.5):
    mask = torch.rand_like(tensor) > mask_prob
    return tensor * mask

data_tensor = add_mask(data_tensor, mask_prob=0.01)