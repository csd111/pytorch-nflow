import unittest
import torch
from layers import BatchReNorm2d


class BatchReNorm2DTest(unittest.TestCase):

    def test_initialization(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        device = torch.device("cpu")
        data = 2 + 3 * torch.randn(2, 8, 32, 32).to(device)
        # ----------------------------------------------------------------------
        # Prepare the layer in standard BatchNorm mode
        # ----------------------------------------------------------------------
        norm = BatchReNorm2d(8, r_max=1, d_max=0)
        norm.to(device)
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        # --------------------------------
        # In training mode
        # --------------------------------
        out = norm(data)
        self.assertLessEqual(torch.mean(out).item(), 5e-8)
        self.assertLessEqual((torch.std(out) - 1).abs().item(), 5e-5)
        # --------------------------------
        # In eval mode
        # --------------------------------
        norm.eval()
        out_eval = norm(data)
        self.assertEqual(torch.mean(torch.abs(out_eval - out)).item(), 0)

    def test_eval_mode(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        device = torch.device("cpu")
        data = 2 + 3 * torch.randn(2, 8, 32, 32).to(device)
        # ----------------------------------------------------------------------
        # Prepare the layer with default init
        # ----------------------------------------------------------------------
        norm = BatchReNorm2d(8)
        norm.to(device)
        norm.eval()
        # ----------------------------------------------------------------------
        # Assess the results stay the same after subsequent calls
        # ----------------------------------------------------------------------
        out = norm(data)
        out2 = norm(data)
        self.assertLessEqual(torch.mean(torch.abs(out2 - out)).item(), 5e-5)


if __name__ == '__main__':
    unittest.main()
