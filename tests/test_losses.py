import unittest
import torch
import math
from training.losses import NLLFlowLoss


class LossTest(unittest.TestCase):

    def test_negative_log_lkl(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.randn(32, 3, 64, 64)
        # ----------------------------------------------------------------------
        # Prepare the loss
        # ----------------------------------------------------------------------
        loss = NLLFlowLoss(sigma=1.0, quantization=256, bits_per_dim=True)
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out = loss(data, torch.zeros(data.size(0)))
        corrected_out = out - math.log(256)/math.log(2)
        # --------------
        self.assertLessEqual(corrected_out, 3)


if __name__ == '__main__':
    unittest.main()
