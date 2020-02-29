import torch
import torch.nn as nn
import torch.nn.init as init
from .normalization import ActNorm2d, BatchReNorm2d

# ------------------------------------------------------------------------------
# Real-NVP
# ------------------------------------------------------------------------------


class RealNVPModule(nn.Module):
    
    def __init__(self, input_channels: int, hidden_channels: int):
        super(RealNVPModule, self).__init__()
        self.conv_in = nn.Conv2d(input_channels, hidden_channels,
                                 kernel_size=3, padding=1, bias=False)
        # --------------
        self.conv_1x1 = nn.Conv2d(hidden_channels, hidden_channels,
                                  kernel_size=1, padding=0, bias=False)
        self.conv_1x1_a = nn.Conv2d(hidden_channels, hidden_channels,
                                    kernel_size=1, padding=0, bias=False)
        self.conv_1x1_b = nn.Conv2d(hidden_channels, hidden_channels,
                                    kernel_size=1, padding=0, bias=False)
        # --------------
        self.norm_1 = BatchReNorm2d(hidden_channels)
        self.conv_1 = nn.Conv2d(hidden_channels, hidden_channels,
                                kernel_size=3, padding=1, bias=False)
        # --------------
        self.norm_2 = BatchReNorm2d(hidden_channels)
        self.conv_2 = nn.Conv2d(hidden_channels, hidden_channels,
                                kernel_size=3, padding=1, bias=False)
        # --------------
        self.norm_3 = BatchReNorm2d(hidden_channels)
        self.conv_3 = nn.Conv2d(hidden_channels, hidden_channels,
                                kernel_size=3, padding=1, bias=False)
        # --------------
        self.norm_4 = BatchReNorm2d(hidden_channels)
        self.conv_4 = nn.Conv2d(hidden_channels, hidden_channels,
                                kernel_size=3, padding=1, bias=False)
        # --------------
        self.norm_out = BatchReNorm2d(hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, 2 * input_channels,
                                  kernel_size=1, padding=0, bias=True)
        # --------------
        init.kaiming_normal_(self.conv_in.weight)
        init.kaiming_normal_(self.conv_1x1_a.weight)
        init.kaiming_normal_(self.conv_1x1_b.weight)
        init.kaiming_normal_(self.conv_1.weight)
        init.kaiming_normal_(self.conv_2.weight)
        init.kaiming_normal_(self.conv_3.weight)
        init.kaiming_normal_(self.conv_4.weight)
        init.zeros_(self.conv_out.weight)
        init.zeros_(self.conv_out.bias)

    def forward(self, x):
        x = self.conv_in(x)
        # -----------------
        x_a = self.norm_1(x)
        x_a = torch.relu(x_a)
        x_a = self.conv_1(x_a)
        # -----------------
        x_a = self.norm_2(x_a)
        x_a = torch.relu(x_a)
        x_a = torch.add(self.conv_2(x_a), x)
        # -----------------
        x_b = self.norm_3(x_a)
        x_b = torch.relu(x_b)
        x_b = self.conv_3(x_b)
        # -----------------
        x_b = self.norm_4(x_b)
        x_b = torch.relu(x_b)
        x_b = torch.add(self.conv_4(x_b), x_a)
        # -----------------
        x = torch.add(self.conv_1x1(x),
                      torch.add(self.conv_1x1_a(x_a), self.conv_1x1_b(x_b)))
        x = self.norm_out(x)
        x = torch.relu(x)
        x = self.conv_out(x)
        return torch.chunk(x, 2, dim=1)

    # --------------------------------------------------------------------------

    def apply_weight_norm(self):
        self.conv_in = nn.utils.weight_norm(self.conv_in)
        self.conv_1 = nn.utils.weight_norm(self.conv_1)
        self.conv_2 = nn.utils.weight_norm(self.conv_2)
        self.conv_3 = nn.utils.weight_norm(self.conv_3)
        self.conv_4 = nn.utils.weight_norm(self.conv_4)
        self.conv_1x1 = nn.utils.weight_norm(self.conv_1x1)
        self.conv_1x1_a = nn.utils.weight_norm(self.conv_1x1_a)
        self.conv_1x1_b = nn.utils.weight_norm(self.conv_1x1_b)

    def remove_weight_norm(self):
        self.conv_in = nn.utils.remove_weight_norm(self.conv_in)
        self.conv_1 = nn.utils.remove_weight_norm(self.conv_1)
        self.conv_2 = nn.utils.remove_weight_norm(self.conv_2)
        self.conv_3 = nn.utils.remove_weight_norm(self.conv_3)
        self.conv_4 = nn.utils.remove_weight_norm(self.conv_4)
        self.conv_1x1 = nn.utils.remove_weight_norm(self.conv_1x1)
        self.conv_1x1_a = nn.utils.remove_weight_norm(self.conv_1x1_a)
        self.conv_1x1_b = nn.utils.remove_weight_norm(self.conv_1x1_b)
    
# ------------------------------------------------------------------------------
# Glow
# ------------------------------------------------------------------------------


class GlowModule(nn.Module):

    def __init__(self, input_channels: int, hidden_channels: int):
        super(GlowModule, self).__init__()
        self.norm_1 = ActNorm2d(hidden_channels)
        self.conv_1 = nn.Conv2d(input_channels + 1, hidden_channels,
                                kernel_size=3, padding=0, bias=False)
        # --------------
        self.norm_2 = ActNorm2d(hidden_channels)
        self.conv_2 = nn.Conv2d(hidden_channels, hidden_channels,
                                kernel_size=1, padding=0, bias=False)
        # --------------
        self.conv_out = nn.Conv2d(hidden_channels + 1, 2 * input_channels,
                                  kernel_size=3, padding=0, bias=True)
        self.log_scale = nn.Parameter(torch.zeros(1, 2 * input_channels, 1, 1),
                                      requires_grad=True)
        # --------------
        init.kaiming_normal_(self.conv_1.weight)
        init.kaiming_normal_(self.conv_2.weight)
        init.zeros_(self.conv_out.weight)
        init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor):
        x = self.conv_1(self._padding(x))
        x, _ = self.norm_1(x)
        x = torch.relu(x)
        # -----------------
        x = self.conv_2(x)
        x, _ = self.norm_2(x)
        x = torch.relu(x)
        # -----------------
        x = self.conv_out(self._padding(x))
        x = x * torch.exp(self.log_scale)
        return torch.chunk(x, 2, dim=1)

    @staticmethod
    def _padding(x: torch.Tensor):
        extra_fm = torch.zeros(x.size(0), 1, x.size(2), x.size(3),
                               device=x.device)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        extra_fm = \
            torch.nn.functional.pad(extra_fm, (1, 1, 1, 1),
                                    mode='constant', value=1)
        return torch.cat((x, extra_fm), dim=1)

    # --------------------------------------------------------------------------

    def apply_weight_norm(self):
        self.conv_1 = nn.utils.weight_norm(self.conv_1)
        self.conv_2 = nn.utils.weight_norm(self.conv_2)

    def remove_weight_norm(self):
        self.conv_1 = nn.utils.remove_weight_norm(self.conv_1)
        self.conv_2 = nn.utils.remove_weight_norm(self.conv_2)
