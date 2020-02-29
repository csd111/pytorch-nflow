import torch
import torch.nn as nn

from .model import _Model
from layers import LogitPreProcess, RealNVPCoupling, \
    Squeeze2x2, AlternateSqueeze2x2

# ------------------------------------------------------------------------------
# External Interface
# ------------------------------------------------------------------------------


class RealNVP(_Model):
    
    """
    Real-NVP normalising flow

    ----------------------------------------------------------------------------
    Ref : Density estimation using Real NVP - https://arxiv.org/abs/1605.08803
    ----------------------------------------------------------------------------

    :param context_blocks: the number of resolutions in the RealNVP model
    :param input_channels: the number of channels of the input tensor
    :param hidden_channels: the number of channels in the affine coupling layers
    :param quantization: the discretization level in the input data
    """

    def __init__(self, context_blocks: int = 2, input_channels: int = 3,
                 hidden_channels: int = 64, quantization: float = 256):
        super(RealNVP, self).__init__()
        self.config = {"name": "RealNVP",
                       "context_blocks": context_blocks,
                       "input_channels": input_channels,
                       "hidden_channels": hidden_channels,
                       "quantization": quantization}
        self.pre_process = \
            LogitPreProcess(scaling=0.9, quantization=quantization)
        self.flows = _RealNVP(0, context_blocks,
                              input_channels, hidden_channels)

    @classmethod
    def from_config(cls, config: dict):
        return cls(context_blocks=config["context_blocks"],
                   input_channels=config["input_channels"],
                   hidden_channels=config["hidden_channels"],
                   quantization=config["quantization"])

    # --------------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        x, log_det_pre = self.pre_process(x)
        x, log_det_flow = self.flows(x)
        return x, log_det_pre + log_det_flow

    def reverse(self, z: torch.Tensor):
        x = self.flows.reverse(z)
        x = self.pre_process.reverse(x)
        return x

    # --------------------------------------------------------------------------

    def apply_weight_norm(self):
        self.flows.apply_weight_norm()

    def remove_weight_norm(self):
        self.flows.remove_weight_norm()

# ------------------------------------------------------------------------------
# Internal recursive model builder
# ------------------------------------------------------------------------------


class _RealNVP(nn.Module):
    """
    Recursive Real-NVP builder class - Each resolution
    """
    def __init__(self, resolution_index: int, nb_resolutions: int,
                 input_channels: int, hidden_channels: int):
        super(_RealNVP, self).__init__()
        # Instantiate the batch of checkerboard affine coupling layers
        self.checkerboard_couplings = nn.ModuleList([
            RealNVPCoupling(input_channels, hidden_channels,
                            checkerboard_mask=True,
                            invert_mask=False),
            RealNVPCoupling(input_channels, hidden_channels,
                            checkerboard_mask=True,
                            invert_mask=True),
            RealNVPCoupling(input_channels, hidden_channels,
                            checkerboard_mask=True,
                            invert_mask=False)])
        # Check if we are building the lowest resolution
        if resolution_index == (nb_resolutions - 1):
            # Final block - no more squeezing - add one last coupling layer
            self.checkerboard_couplings.append(
                RealNVPCoupling(input_channels, hidden_channels,
                                checkerboard_mask=True,
                                invert_mask=True))
            # --------------------
            self.last_block = True
        else:
            # Instantiate the batch of channel wise affine coupling layers
            self.channel_couplings = nn.ModuleList([
                RealNVPCoupling(4 * input_channels, 2 * hidden_channels,
                                checkerboard_mask=False,
                                invert_mask=False),
                RealNVPCoupling(4 * input_channels, 2 * hidden_channels,
                                checkerboard_mask=False,
                                invert_mask=True),
                RealNVPCoupling(4 * input_channels, 2 * hidden_channels,
                                checkerboard_mask=False,
                                invert_mask=False)])
            # Instantiate the squeeze layers
            self.squeeze = Squeeze2x2()
            self.alt_squeeze = AlternateSqueeze2x2()
            # Instantiate next context block (i.e. resolution) - recursive call
            self.next_block = _RealNVP(resolution_index + 1, nb_resolutions,
                                       2 * input_channels, 2 * hidden_channels)
            # ---------------------
            self.last_block = False

    # --------------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        # Instantiate the sum of individual log-determinants
        sum_log_det = torch.zeros(x.size(0), device=x.device)
        # ----------------------------------------------------------------------
        # Checkerboard affine coupling layers
        # ----------------------------------------------------------------------
        for coupling in self.checkerboard_couplings:
            x, log_det = coupling(x)
            sum_log_det += log_det
        # Unless we have reached the bottom, continue
        if not self.last_block:
            # ------------------------------------------------------------------
            # Squeeze -> channel-wise affine coupling layers -> Unsqueeze
            # ------------------------------------------------------------------
            x, _ = self.squeeze(x)
            for coupling in self.channel_couplings:
                x, log_det = coupling(x)
                sum_log_det += log_det
            x = self.squeeze.reverse(x)
            # ------------------------------------------------------------------
            # Alternate squeeze -> Split |-> Next Context Block |-> Unsqueeze
            # ------------------------------------------------------------------
            x, _ = self.alt_squeeze(x)
            x, x_factored_out = x.chunk(2, dim=1)
            x, log_det = self.next_block(x)
            sum_log_det += log_det
            x = torch.cat((x, x_factored_out), dim=1)
            x = self.alt_squeeze.reverse(x)
        return x, sum_log_det

    def reverse(self, z: torch.Tensor):
        if not self.last_block:
            # ------------------------------------------------------------------
            z, _ = self.alt_squeeze(z)
            z, z_factored_out = z.chunk(2, dim=1)
            z = self.next_block.reverse(z)
            z = torch.cat((z, z_factored_out), dim=1)
            z = self.alt_squeeze.reverse(z)
            # ------------------------------------------------------------------
            z, _ = self.squeeze(z)
            for coupling in reversed(self.channel_couplings):
                z = coupling.reverse(z)
            z = self.squeeze.reverse(z)
        # ----------------------------------------------------------------------
        for coupling in reversed(self.checkerboard_couplings):
            z = coupling.reverse(z)
        return z

    # --------------------------------------------------------------------------

    def apply_weight_norm(self):
        for coupling in self.checkerboard_couplings:
            coupling.apply_weight_norm()
        if not self.last_block:
            self.next_block.apply_weight_norm()
            for coupling in self.channel_couplings:
                coupling.apply_weight_norm()

    def remove_weight_norm(self):
        for coupling in self.checkerboard_couplings:
            coupling.remove_weight_norm()
        if not self.last_block:
            self.next_block.remove_weight_norm()
            for coupling in self.channel_couplings:
                coupling.remove_weight_norm()
