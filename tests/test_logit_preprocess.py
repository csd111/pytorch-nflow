import unittest
import torch
import math
import torch.nn.functional as F
from layers import LogitPreProcess


class Invertible1x1ConvTest(unittest.TestCase):

    def test_invertibility(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.rand(2, 3, 64, 64)
        # ----------------------------------------------------------------------
        # Prepare the layer
        # ----------------------------------------------------------------------
        logit_pre = LogitPreProcess(scaling=0.9, quantization=65536)
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = logit_pre(data)
        back = logit_pre.reverse(out)
        x = (0.9 * (2 * data - 1) + 1) / 2
        mapped_x = torch.log(x) - torch.log(1 - x)
        pre_logit_scale = torch.Tensor([math.log(0.9) - math.log(1 - 0.9)])
        target_log_det = \
            (F.softplus(mapped_x) + F.softplus(- mapped_x) - F.softplus(
                -pre_logit_scale)).flatten(1).sum(-1)
        # -------------
        error_reco = torch.mean(torch.abs(back - data)).item()
        error_log_det = torch.mean(torch.abs(log_det - target_log_det) /
                                   torch.abs(log_det)).item()
        # The errors directly depend on the amount of jittering applied
        self.assertLessEqual(error_reco, 1e-5)
        self.assertLessEqual(error_log_det, 5e-5)
        # Test the layer changed the space from [0, 1] to unbounded space
        self.assertGreaterEqual(torch.min(data), 0)
        self.assertLessEqual(torch.max(data), 1)
        # ------
        self.assertLessEqual(torch.min(out), 0)
        self.assertGreaterEqual(torch.max(out), 1)


if __name__ == '__main__':
    unittest.main()
