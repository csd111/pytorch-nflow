import torch
import torch.nn as nn

from .model import _Model
from layers import LogitPreProcess, GlowCoupling, ActNorm2d, \
    Squeeze2x2, Invertible1x1Conv


# ------------------------------------------------------------------------------
# External Interface
# ------------------------------------------------------------------------------


class Glow(_Model):
    """
    Glow : Generative Flow with Invertible 1x1 Convolutions

    ----------------------------------------------------------------------------
    Ref : Glow paper - https://arxiv.org/abs/1807.03039
    ----------------------------------------------------------------------------

    :param context_blocks: the number of resolutions in the Glow model
    :param flow_steps: the number of steps of flow at each resolution
    :param input_channels: the number of channels of the input tensor
    :param hidden_channels: the number of channels in the affine coupling layers
    :param quantization: the discretization level in the input data
     replaced by stacking along the batch dimension, simulating weight sharing
    :param lu_decomposition: whether to use lu decomposition in the invertible
     1x1 convolution
    """

    def __init__(self, context_blocks: int = 2, flow_steps: int = 3, 
                 input_channels: int = 3, hidden_channels: int = 64,
                 quantization: float = 256, lu_decomposition: bool = False):
        super(Glow, self).__init__()
        self.config = {"name": "Glow",
                       "context_blocks": context_blocks,
                       "flow_steps": flow_steps,
                       "input_channels": input_channels,
                       "hidden_channels": hidden_channels,
                       "quantization": quantization,
                       "lu_decomposition": lu_decomposition}
        self.squeeze = Squeeze2x2()
        self.pre_process = \
            LogitPreProcess(scaling=0.9, quantization=quantization)
        self.flows = _Glow(context_blocks, flow_steps,
                           4 * input_channels, hidden_channels,
                           lu_decomposition)

    @classmethod
    def from_config(cls, config: dict):
        return cls(context_blocks=config["context_blocks"],
                   flow_steps=config["flow_steps"],
                   input_channels=config["input_channels"],
                   hidden_channels=config["hidden_channels"],
                   quantization=config["quantization"],
                   lu_decomposition=config["lu_decomposition"])

    # --------------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        x, log_det_pre = self.pre_process(x)
        x, _ = self.squeeze(x)
        x, log_det_flow = self.flows(x)
        x = self.squeeze.reverse(x)
        return x, log_det_pre + log_det_flow

    def reverse(self, z: torch.Tensor):
        x, _ = self.squeeze(z)
        x = self.flows.reverse(x)
        x = self.squeeze.reverse(x)
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


class _Glow(nn.Module):
    """
    Recursive Glow builder class - Each resolution
    """

    def __init__(self, nb_resolutions: int, nb_flows_steps: int,
                 input_channels: int, hidden_channels: int,
                 lu_decomposition: bool):
        super(_Glow, self).__init__()
        # Instantiate all the flow steps at this resolution
        self.flows = nn.ModuleList(
            [_FlowStep(input_channels, hidden_channels, lu_decomposition)
             for _ in range(nb_flows_steps)])
        # Check if we are building the last resolution context block
        if nb_resolutions > 1:
            self.next_block = _Glow(nb_resolutions - 1, nb_flows_steps,
                                    2 * input_channels, 2 * hidden_channels,
                                    lu_decomposition)
            self.last_block = False
        else:
            self.last_block = True
        # Instantiate the squeeze layer
        self.squeeze = Squeeze2x2()

    # --------------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        # Instantiate the sum of individual log-determinants
        sum_log_det = torch.zeros(x.size(0), device=x.device)
        # ----------------------------------------------------------------------
        for flow in self.flows:
            x, log_det = flow(x)
            sum_log_det += log_det
        # Unless we have reached the bottom, continue
        if not self.last_block:
            # ------------------------------------------------------------------
            x, _ = self.squeeze(x)
            x, x_factored_out = x.chunk(2, dim=1)
            x, log_det = self.next_block(x)
            sum_log_det += log_det
            x = torch.cat((x, x_factored_out), dim=1)
            x = self.squeeze.reverse(x)
        return x, sum_log_det

    def reverse(self, z: torch.Tensor):
        if not self.last_block:
            # ------------------------------------------------------------------
            z, _ = self.squeeze(z)
            z, z_factored_out = z.chunk(2, dim=1)
            z = self.next_block.reverse(z)
            z = torch.cat((z, z_factored_out), dim=1)
            z = self.squeeze.reverse(z)
        # ----------------------------------------------------------------------
        for flow in reversed(self.flows):
            z = flow.reverse(z)
        return z

    # --------------------------------------------------------------------------

    def apply_weight_norm(self):
        for flow in self.flows:
            flow.apply_weight_norm()
        if not self.last_block:
            self.next_block.apply_weight_norm()

    def remove_weight_norm(self):
        for flow in self.flows:
            flow.remove_weight_norm()
        if not self.last_block:
            self.next_block.remove_weight_norm()

# ------------------------------------------------------------------------------


class _FlowStep(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, lu: bool):
        super(_FlowStep, self).__init__()
        self.act_norm = ActNorm2d(input_channels)
        self.inv_1x1_conv = Invertible1x1Conv(input_channels,
                                              lu_decomposition=lu)
        self.coupling = GlowCoupling(input_channels, hidden_channels)

    def forward(self, x: torch.Tensor):
        x, log_det_a = self.act_norm(x)
        x, log_det_b = self.inv_1x1_conv(x)
        x, log_det_c = self.coupling(x)
        return x, log_det_a + log_det_b + log_det_c

    def reverse(self, z: torch.Tensor):
        z = self.coupling.reverse(z)
        z = self.inv_1x1_conv.reverse(z)
        z = self.act_norm.reverse(z)
        return z

    # --------------------------------------------------------------------------

    def apply_weight_norm(self):
        self.coupling.apply_weight_norm()

    def remove_weight_norm(self):
        self.coupling.remove_weight_norm()
