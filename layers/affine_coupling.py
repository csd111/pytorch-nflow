import torch
import torch.nn as nn
from enum import Enum
from .squeeze import squeeze2x2_alternate, unsqueeze2x2_alternate
from .coupling_modules import RealNVPModule, GlowModule

# ------------------------------------------------------------------------------
# Mask Type determining how to perform the splitting
# ------------------------------------------------------------------------------


class MaskType(Enum):
    CHANNEL_WISE = 0
    CHECKERBOARD = 1
    
# ------------------------------------------------------------------------------
# Affine Coupling Base Class
# ------------------------------------------------------------------------------


class AffineCoupling(nn.Module):
    """
    Affine Coupling layer base class - Implements the general behaviour of all
    affine coupling layers, just the coupling module needs to be defined
    externally. Like the original paper, log(s) is parameterised through 
    [scale * tanh(net_out)] for stability. Please note that in the original 
    Open AI Glow code, this is done using [scale * sigmoid(net_out + 2)].

    Reference : See Real-NVP paper for details https://arxiv.org/abs/1605.08803
    
    :param coupling_module: the neural network module which computes t & log(s)
    :param mask_type: how to perform the splitting operation, i.e. checkerboard 
     or channel-wise
    :param invert_mask: whether to reverse the input halves for split and concat
     operations (equivalent to inverting the mask)
    """
    def __init__(self, coupling_module: nn.Module,
                 nb_channels: int,
                 mask_type: MaskType = MaskType.CHANNEL_WISE,
                 invert_mask: bool = False):
        super(AffineCoupling, self).__init__()
        self.__split = mask_type
        self.__invert = invert_mask
        self.nn = coupling_module
        self.scale = nn.Parameter(torch.ones(1, nb_channels, 1, 1),
                                  requires_grad=True)
        
    # --------------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        # Split the input tensor in two halves
        x_id, x_m = self._split(x)
        # Go through the coupling module to get log(s) and t
        s, t = self.nn(x_id)
        log_s = self.scale * torch.tanh(s)
        # Perform translation and scaling
        x_m = x_m * torch.exp(log_s) + t
        # Re-assemble the two halves
        x = self._concat(x_id, x_m)
        # Compute the log-determinant of the Jacobian
        log_det = log_s.ﬂatten(1).sum(-1)
        return x, log_det
    
    def reverse(self, z: torch.Tensor):
        # Split the input tensor in two halves
        z_id, z_m = self._split(z)
        # Go through the coupling module to get log(s) and t
        s, t = self.nn(z_id)
        log_s = self.scale * torch.tanh(s)
        # Perform inverse translation and scaling
        z_m = (z_m - t) * torch.exp(-log_s)
        # Re-assemble the two halves
        return self._concat(z_id, z_m)

    # --------------------------------------------------------------------------
    
    def _split(self, x: torch.Tensor):
        """
        Performs the appropriate splitting operation depending on the type of
        mask and whether it should be inverted.

        """
        if self.__split == MaskType.CHECKERBOARD:
            # Checkerboard masking via squeeze with alternated channels
            x = squeeze2x2_alternate(x)
        # Split the channels
        x_id, x_m = torch.chunk(x, 2, dim=1)
        # ------------------------
        if self.__invert:
            return x_m, x_id
        else:
            return x_id, x_m

    def _concat(self, x_1: torch.Tensor, x_2: torch.Tensor):
        """
        Performs the appropriate concatenation operation depending on the type
        of mask and whether it should be inverted.

        """
        if self.__invert:
            x_1, x_2 = x_2, x_1
        # ------------------------
        x = torch.cat((x_1, x_2), dim=1)
        if self.__split == MaskType.CHECKERBOARD:
            # Un-squeeze the alternated channels
            x = unsqueeze2x2_alternate(x)
        return x

    # --------------------------------------------------------------------------
    
    def apply_weight_norm(self):
        self.nn.apply_weight_norm()

    def remove_weight_norm(self):
        self.nn.remove_weight_norm()


# ------------------------------------------------------------------------------
# Affine Coupling Children
# ------------------------------------------------------------------------------

class RealNVPCoupling(AffineCoupling):
    
    def __init__(self, input_channels: int, hidden_channels: int, 
                 checkerboard_mask: bool = False,
                 invert_mask: bool = False):
        # ----------------------------------------------------------------------
        if checkerboard_mask is True:
            mask_type = MaskType.CHECKERBOARD
            in_channels = input_channels * 2
            mid_channels = hidden_channels * 2
        else:
            mask_type = MaskType.CHANNEL_WISE
            in_channels = input_channels // 2
            mid_channels = hidden_channels // 2
        # ----------------------------------------------------------------------
        network = RealNVPModule(in_channels, mid_channels)
        super(RealNVPCoupling, self).__init__(network, in_channels,
                                              mask_type, invert_mask)


class GlowCoupling(AffineCoupling):

    def __init__(self, input_channels: int, hidden_channels: int,
                 checkerboard_mask: bool = False,
                 invert_mask: bool = False):
        # ----------------------------------------------------------------------
        if checkerboard_mask is True:
            mask_type = MaskType.CHECKERBOARD
            in_channels = input_channels * 2
            mid_channels = hidden_channels * 2
        else:
            mask_type = MaskType.CHANNEL_WISE
            in_channels = input_channels // 2
            mid_channels = hidden_channels // 2
        # ----------------------------------------------------------------------
        network = GlowModule(in_channels, mid_channels)
        super(GlowCoupling, self).__init__(network, in_channels,
                                           mask_type, invert_mask)
