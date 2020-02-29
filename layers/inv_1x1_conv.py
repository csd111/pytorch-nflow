import torch
import torch.nn as nn
import torch.nn.functional as f


class Invertible1x1Conv(nn.Module):

    def __init__(self, nb_channels: int, lu_decomposition: bool = False):
        """
        Invertible 1x1 Convolution for 2D inputs with LU parameterization.
        References : See Glow paper for details https://arxiv.org/abs/1807.03039

        :param nb_channels: the number of input/output channels
        :param lu_decomposition: whether to use LU parameterization
        """
        super(Invertible1x1Conv, self).__init__()

        self.__channels = nb_channels
        # Initialize W as a random orthogonal matrix
        W = torch.zeros(nb_channels, nb_channels)
        nn.init.orthogonal_(W)
        # Make sure the det is 1 (and not -1) to have a defined log-det
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        self.__lu = lu_decomposition
        if self.__lu:
            W_LU, pivots = W.lu()
            P, L, U = torch.lu_unpack(W_LU, pivots)
            s = torch.diag(U)
            U = U - torch.diag(s)
            # Assign the module trainable parameters
            self.sign_s = nn.Parameter(torch.sign(s), requires_grad=True)
            self.log_s = nn.Parameter(torch.log(s.abs()), requires_grad=True)
            self.lower = nn.Parameter(L, requires_grad=True)
            self.upper = nn.Parameter(U, requires_grad=True)
            # Assign the non trainable ones
            self.register_buffer('permutation', P)
            self.register_buffer('permutation_inv', torch.inverse(P))
            self.register_buffer('eye', torch.eye(nb_channels))
            self.register_buffer('l_mask', torch.tril(torch.ones(
                nb_channels, nb_channels), -1))
        else:
            self.weight = nn.Parameter(W, requires_grad=True)

    def forward(self, x: torch.Tensor):
        if self.__lu:
            # Reconstruct the weight matrix
            s_mat = torch.diag(torch.exp(self.log_s) * torch.sign(self.sign_s))
            u_mat = self.upper * self.l_mask.t() + s_mat
            l_mat = self.lower * self.l_mask + self.eye
            weight = torch.matmul(self.permutation, torch.matmul(l_mat, u_mat))
            # Compute the log-determinant of the Jacobian
            log_det = torch.sum(self.log_s) * x.size(2) * x.size(3)
        else:
            weight = self.weight
            log_det = torch.logdet(weight) * x.size(2) * x.size(3)
        # Run the convolution
        z = f.conv2d(x, weight.view(self.__channels, self.__channels, 1, 1))
        return z, log_det.expand(x.size(0))

    def reverse(self, z: torch.Tensor):
        if self.__lu:
            # Reconstruct the inverse weight matrix
            s_mat = torch.diag(torch.exp(self.log_s) * torch.sign(self.sign_s))
            u_inv = torch.inverse(self.upper * self.l_mask.t() + s_mat)
            l_inv = torch.inverse(self.lower * self.l_mask + self.eye)
            weight = torch.matmul(u_inv,
                                  torch.matmul(l_inv, self.permutation_inv))
        else:
            weight = torch.inverse(self.weight)
        # Run the convolution
        x = f.conv2d(z, weight.view(self.__channels, self.__channels, 1, 1))
        return x
