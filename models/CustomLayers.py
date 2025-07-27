import numpy
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x *