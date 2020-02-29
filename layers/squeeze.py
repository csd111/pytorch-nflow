import torch
import torch.nn as nn

# ------------------------------------------------------------------------------
# Module interface
# ------------------------------------------------------------------------------


class Squeeze2x2(nn.Module):
    
    def __init__(self):
        super(Squeeze2x2, self).__init__()
        
    def forward(self, x: torch.Tensor):
        return squeeze2x2(x), torch.zeros(1, device=x.device)

    def reverse(self, z: torch.Tensor):
        return unsqueeze2x2(z)


class AlternateSqueeze2x2(nn.Module):

    def __init__(self):
        super(AlternateSqueeze2x2, self).__init__()

    def forward(self, x: torch.Tensor):
        return squeeze2x2_alternate(x), torch.zeros(1, device=x.device)

    def reverse(self, z: torch.Tensor):
        return unsqueeze2x2_alternate(z)

# ------------------------------------------------------------------------------
# Functional interface
# ------------------------------------------------------------------------------


def squeeze2x2(x: torch.Tensor):
    b, c, h, w = x.size()
    out = x.view(-1, c, h // 2, 2, w // 2, 2)
    out = out.permute(0, 1, 3, 5, 2, 4)
    return out.contiguous().view(-1, c * 4, h // 2, w // 2)


def unsqueeze2x2(x: torch.Tensor):
    b, c, h, w = x.size()
    out = x.view(-1, c // 4, 2, 2, h, w)
    out = out.permute(0, 1, 4, 2, 5, 3)
    return out.contiguous().view(-1, c // 4, 2 * h, 2 * w)


def squeeze2x2_alternate(x: torch.Tensor):
    b, c, h, w = x.size()
    out = x.view(-1, c, h // 2, 2, w // 2, 2)
    out = out.permute(0, 1, 3, 5, 2, 4)
    out = out.contiguous().view(-1, c, 4, h // 2, w // 2)
    return out[:, :, [0, 3, 1, 2], :, :].view(-1, c * 4, h // 2, w // 2)


def unsqueeze2x2_alternate(x: torch.Tensor):
    b, c, h, w = x.size()
    out = x.view(-1, c // 4, 4, h, w)
    out = out[:, :, [0, 2, 3, 1], :, :].view(-1, c // 4, 2, 2, h, w)
    out = out.permute(0, 1, 4, 2, 5, 3)
    return out.contiguous().view(-1, c // 4, 2 * h, 2 * w)
