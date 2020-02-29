import math
import torch
import torch.nn as nn


class LogitPreProcess(nn.Module):

    def __init__(self, scaling: float = 0.9, quantization: float = 256.0):
        """
        Adds jittering to dequantize the input tensor and maps the values from 
        [0, 1] to unbounded space to reduce boundary effects.
        References : See https://arxiv.org/abs/1511.01844 and 
        https://arxiv.org/abs/1306.0186 for the jittering procedure, and the 
        Real-NVP paper https://arxiv.org/abs/1605.08803 for the logit mapping.

        :param scaling: the amount of shrinking applied before the logit
        :param quantization: the discretization level in the input data
        """
        super(LogitPreProcess, self).__init__()
        self.__scaling = scaling
        self.__disc_lvl = quantization

    def forward(self, x: torch.Tensor):
        # De-quantize the data
        x = (x * (self.__disc_lvl - 1) + torch.rand_like(x)) / self.__disc_lvl
        # Scale to contract inside [0, 1]
        x = ((2 * x - 1) * self.__scaling + 1) / 2
        # Apply logit to map to unbounded space
        mapped_x = torch.log(x) - torch.log(1 - x)
        # Compute contribution to the log-likelihood
        log_det = - torch.log(x) - torch.log(1 - x) + math.log(self.__scaling)
        return mapped_x, log_det.flatten(1).sum(-1)

    def reverse(self, z: torch.Tensor):
        # Reverse the logit
        z = 1 / (1 + torch.exp(-z))
        # Reverse the scaling operation
        x = ((2 * z - 1) / self.__scaling + 1) / 2
        return x
