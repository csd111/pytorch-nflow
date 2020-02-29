import torch
import torch.nn as nn

# ------------------------------------------------------------------------------
# ActNorm2d
# ------------------------------------------------------------------------------


class ActNorm2d(nn.Module):

    def __init__(self, nb_channels: int, scale: float = 2., eps: float = 1e-3):
        """
        Activation normalization for 2D inputs. !! Warning !! - Data-dependent 
        initialization only works if the module is in training mode - Avoid use
        right after a layer initialized to give 0 output, as it might make the
        following batch blow up
        
        References : See Glow paper for details https://arxiv.org/abs/1807.03039

        :param nb_channels: the number of input/output channels
        :param scale: scale applied to the standard deviation during data-
         -dependent initialization
        :param eps: a value added to the denominator for numerical stability
        """
        super(ActNorm2d, self).__init__()
        self.__scale = scale
        self.__eps = eps
        # Register trainable parameters (the scale is saved as log_s for
        self.bias = nn.Parameter(torch.zeros(1, nb_channels, 1, 1),
                                 requires_grad=True)
        self.log_s = nn.Parameter(torch.zeros(1, nb_channels, 1, 1),
                                  requires_grad=True)
        # Register the internal state for initialization
        self.register_buffer('init', torch.zeros(1))
        
    def forward(self, x: torch.Tensor):
        # If first batch, run data-dependent initialization
        if self.init.item() == 0:
            self._initialize_parameters(x)
        # Now translate and scale along the channels dimension
        z = (x + self.bias) * torch.exp(self.log_s)
        # Compute the log-determinant of the Jacobian
        log_det = torch.sum(self.log_s) * x.size(2) * x.size(3)
        return z, log_det.expand(x.size(0))
    
    def reverse(self, z: torch.Tensor):
        # Invert the translation and scaling
        x = z * torch.exp(- self.log_s) - self.bias
        return x
    
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    def _initialize_parameters(self, x: torch.Tensor):
        if self.training:
            with torch.no_grad():
                mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
                variance = torch.mean((x - mean) ** 2,
                                      dim=[0, 2, 3], keepdim=True)
                log_s = - 0.5 * self.__scale * torch.log(variance + self.__eps)
                self.bias.data.copy_(- mean.data)
                self.log_s.data.copy_(log_s.data)
                self.init = self.init + 1

# ------------------------------------------------------------------------------
# BatchReNorm2d
# ------------------------------------------------------------------------------

                
class BatchReNorm2d(nn.Module):

    def __init__(self, nb_channels: int, alpha: float = 0.05, eps: float = 1e-5,
                 r_max: float = 3, d_max: float = 5):
        """
        Custom batch normalization for 2D inputs as described in the Real-NVP 
        paper and akin to Batch Re-Normalization - should be less sensitive
        to small batch sizes than standard batch norm

        References : - Real-NVP paper https://arxiv.org/abs/1807.03039
        - Batch Re-Normalization paper https://arxiv.org/abs/1702.03275

        :param nb_channels: the number of input/output channels
        :param alpha: moving average update rate
        :param eps: a value added to the denominator for numerical stability
        :param r_max: clamping value for the moving average factor r (set to 1
         for standard BatchNorm behaviour)
        :param d_max: clamping value for the moving average factor d (set to 0
         for standard BatchNorm behaviour)
        """
        super(BatchReNorm2d, self).__init__()
        self.__alpha = alpha
        self.__eps = eps
        self.__r_max = r_max
        self.__d_max = d_max
        # Register the running means and vars
        self.register_buffer('mean', torch.zeros(1, nb_channels, 1, 1))
        self.register_buffer('std_dev', torch.ones(1, nb_channels, 1, 1))

    def forward(self, x: torch.Tensor):
        # Compute batch statistics
        mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
        std = torch.sqrt(torch.mean(
            (x - mean) ** 2, dim=[0, 2, 3], keepdim=True) + self.__eps)
        with torch.no_grad():
            # Compute moving average factors
            r = torch.clamp(
                std / self.std_dev, 1 / self.__r_max, self.__r_max)
            d = torch.clamp(
                (mean - self.mean) / std, - self.__d_max, self.__d_max)
            if self.training:
                # Update moving average estimates
                self.std_dev += self.__alpha * (std - self.std_dev)
                self.mean += self.__alpha * (mean - self.mean)
        # Return the normalized data
        return ((x - mean) / std) * r + d
