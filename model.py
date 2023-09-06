import torch.nn as nn
import torch


class PinSAGE(nn.Module):
    def __init__(self, hidden_dim, num_layer):
        super().__init__()
        
        