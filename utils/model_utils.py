import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

def sample_Z(m, n):
    return torch.rand(m, n)

def sample_M(batch_size, data_shape, p):
    """
    Generate a hint matrix.
    Parameters:
    - batch_size: Number of samples in the batch.
    - data_shape: Shape of each data sample (excluding the batch dimension).
    - p: Probability of hiding an observed data point.
    Returns:
    - Hint matrix with shape (batch_size, *data_shape).
    """
    random_matrix = np.random.uniform(0., 1., size = (batch_size, *data_shape))
    return (random_matrix > p).astype(float)


def get_activation_function(name):
    if hasattr(nn, name):
        return getattr(nn, name)()
    else:
        raise ValueError(f"Unknown activation function: {name}")

def get_activation_function(name):
    if name == "ReLU":
        return nn.ReLU()
    elif name == "LeakyReLU":
        return nn.LeakyReLU(0.2, inplace=True)
    elif name == "Sigmoid":
        return nn.Sigmoid()
    elif name == "Tanh":
        return nn.Tanh()
    elif name == 'PReLU':
        return nn.PReLU()
    else:
        raise ValueError("Unknown activation function")


def get_optimizer(optimizer_name, parameters, lr, weight_decay=0, betas=(0.9, 0.999), momentum=0):
    if optimizer_name == "Adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_name == "SGD":
        return optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == "AdamW":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        raise ValueError(f"Unsupported optimizer type provided: {optimizer_name}")
