import unittest
import torch
import torch.nn.init as init
from layers import Invertible1x1Conv


class Invertible1x1ConvTest(unittest.TestCase):

    def test_invertibility_base_version(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.randn(2, 8, 32, 32)
        # ----------------------------------------------------------------------
        # Prepare the layer
        # ----------------------------------------------------------------------
        inv_conv = Invertible1x1Conv(8, lu_decomposition=False)
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = inv_conv(data)
        back = inv_conv.reverse(out)
        # --------------
        error_reco = torch.mean(torch.abs(back - data)).item()
        self.assertLessEqual(error_reco, 2e-7)

    def test_lu_version(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.randn(2, 8, 32, 32)
        # ----------------------------------------------------------------------
        # Prepare the layer
        # ----------------------------------------------------------------------
        inv_conv = Invertible1x1Conv(8, lu_decomposition=True)
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = inv_conv(data)
        back = inv_conv.reverse(out)
        # --------------
        s_mat = \
            torch.diag(torch.exp(inv_conv.log_s) * inv_conv.sign_s)
        weight_mat = torch.matmul(inv_conv.permutation, 
            torch.matmul(inv_conv.lower, inv_conv.upper + s_mat))
        target_log_det = \
            (torch.logdet(weight_mat) * data.size(2) * data.size(3)).expand(2)
        # --------------
        error_reco = torch.mean(torch.abs(back - data)).item()
        error_log_det = torch.mean(torch.abs(log_det - target_log_det)).item()
        self.assertLessEqual(error_reco, 2e-7)
        self.assertLessEqual(error_log_det, 4e-4)

    def test_lu_weight_decomposition(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.randn(2, 16, 32, 32)
        weight_mat = 2 * torch.eye(16, 16)
        # ----------------------------------------------------------------------
        # Prepare the layer
        # ----------------------------------------------------------------------
        inv_conv = Invertible1x1Conv(16, lu_decomposition=True)
        # initialize weights to get W = I
        init.eye_(inv_conv.permutation)
        init.eye_(inv_conv.lower)
        init.zeros_(inv_conv.upper)
        init.constant_(inv_conv.log_s, 0.6931471805599453)
        init.constant_(inv_conv.sign_s, 1)
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = inv_conv(data)
        target_log_det = \
            (torch.logdet(weight_mat) * data.size(2) * data.size(3)).expand(2)
        # --------------
        error_weight = torch.mean(torch.abs(out - 2 * data)).item()
        error_log_det = torch.mean(torch.abs(log_det - target_log_det)).item()
        self.assertLessEqual(error_weight, 1e-12)
        self.assertLessEqual(error_log_det, 1e-12)


if __name__ == '__main__':
    unittest.main()
