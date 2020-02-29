import unittest
import torch
from layers import ActNorm2d


class ActNorm2DTest(unittest.TestCase):

    def test_invertibility(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.randn(2, 8, 32, 32)
        # ----------------------------------------------------------------------
        # Prepare the layer
        # ----------------------------------------------------------------------
        act_norm = ActNorm2d(8)
        act_norm.init = act_norm.init + 1
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = act_norm(data)
        back = act_norm.reverse(out)
        # --------------------------------
        error_reco = torch.mean(torch.abs(back - data)).item()
        error_out = torch.mean(torch.abs(out - data)).item()
        self.assertLessEqual(error_reco, 2e-7)
        self.assertLessEqual(error_out, 2e-7)
        self.assertEqual(log_det.sum().item(), 0)
        # ----------------------------------------------------------------------
        # Randomize the layer weights
        # ----------------------------------------------------------------------
        act_norm.bias.data.copy_(torch.randn(1, 8, 1, 1))
        act_norm.log_s.data.copy_(6 * torch.randn(1, 8, 1, 1))
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = act_norm(data)
        back = act_norm.reverse(out)
        # --------------------------------
        error_reco = torch.mean(torch.abs(back - data)).item()
        self.assertLessEqual(error_reco, 2e-7)
        # --------------------------------
        out, log_det = act_norm(data)
        back = act_norm.reverse(out)
        # --------------------------------
        error_reco = torch.mean(torch.abs(back - data)).item()
        self.assertLessEqual(error_reco, 2e-7)

    def test_data_dependent_initialization(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        device = torch.device("cpu")
        data = torch.randn(2, 8, 32, 32).to(device)
        # ----------------------------------------------------------------------
        # Prepare the layer
        # ----------------------------------------------------------------------
        act_norm = ActNorm2d(8, scale=1.)
        act_norm.to(device)
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        self.assertEqual(act_norm.init.item(), 0)
        # --------------------------------
        out, log_det = act_norm(data)
        back = act_norm.reverse(out)
        # --------------------------------
        error_reco = torch.mean(torch.abs(back - data)).item()
        self.assertLessEqual(error_reco, 2e-7)
        # --------------------------------
        self.assertLessEqual(torch.mean(out).item(), 2e-8)
        self.assertLessEqual((torch.std(out) - 1).abs().item(), 5e-4)
        self.assertEqual(act_norm.init.item(), 1)


if __name__ == '__main__':
    unittest.main()
