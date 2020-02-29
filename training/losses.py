import warnings
import torch
import math
from numpy import prod

# ------------------------------------------------------------------------------
# Negative Log Likelihood loss
# ------------------------------------------------------------------------------


class NLLFlowLoss(torch.nn.Module):

    def __init__(self, sigma: float = 1.0, quantization: int = 256,
                 bits_per_dim: bool = True):
        """
        Negative log-likelihood for generative flow networks assuming a
        zero-mean gaussian prior. Computed according to (2) in  Kingma et al.
        Glow paper : https://arxiv.org/abs/1807.03039

        :param sigma: the standard deviation of the Gaussian prior
        :param quantization: the discretization level in the input data
        :param bits_per_dim: whether to convert the negative log likelihood loss
         to compression in bits per dimension
        """
        super(NLLFlowLoss, self).__init__()
        self.__disc_lvl = quantization
        self.__sigma_sq = sigma ** 2
        self.__constant = math.log(self.__sigma_sq * 2 * math.pi)
        self.__compression = bits_per_dim

    def forward(self, latent_var: torch.Tensor, sum_log_det: torch.Tensor):
        # Compute the log-prob of the latent according to the Gaussian prior
        log_prob_prior = -0.5 * (latent_var.pow(2).div(self.__sigma_sq) +
                                 self.__constant)
        # Assume iid samples (i.e. pixels) so sum along all dims except batch
        log_prob_prior = torch.sum(
            log_prob_prior.view(log_prob_prior.size(0), -1), -1)
        # Check dimensions are good
        if not (log_prob_prior.size() == sum_log_det.size()):
            warnings.warn(
                "Using a log determinant size ({}) that is different"
                " from the size of the log-likelihood of the latent ({}). "
                "This will likely lead to incorrect "
                "results due to broadcasting. "
                "Please ensure they "
                "have the same size.".format(sum_log_det.size(),
                                             log_prob_prior.size()),
                stacklevel=2)
        # Compute the negative log-likelihood objective to minimize
        data_dim = int(prod(latent_var.size()[1:]))
        n_log_lkl = - torch.mean(log_prob_prior + sum_log_det) + \
            math.log(self.__disc_lvl) * data_dim
        # Check if we want to return bits per dim
        if self.__compression:
            n_log_lkl = n_log_lkl / (math.log(2) * data_dim)
        return n_log_lkl
