import unittest
import torch
import torch.nn.init as init
from layers import RealNVPCoupling, GlowCoupling


class AffineCouplingTest(unittest.TestCase):

    def test_invertibility_real_nvp(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.rand(2, 8, 32, 32)
        # ----------------------------------------------------------------------
        # Prepare the layer with default init
        # ----------------------------------------------------------------------
        coupling = RealNVPCoupling(8, 16,
                                   checkerboard_mask=True, invert_mask=True)
        coupling.eval()
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = coupling(data)
        back = coupling.reverse(out)
        error_reco = torch.mean(torch.abs(back - data)).item()
        # -------------
        self.assertEqual(error_reco, 0)
        self.assertEqual(log_det.sum().item(), 0)
        # ----------------------------------------------------------------------
        # Randomise the weights of the layer
        # ----------------------------------------------------------------------
        init.kaiming_normal_(coupling.nn.conv_out.weight)
        init.ones_(coupling.nn.conv_out.bias)
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = coupling(data)
        back = coupling.reverse(out)
        error_reco = torch.mean(torch.abs(back - data)).item()
        # -------------
        self.assertLessEqual(error_reco, 1e-5)

    def test_invertibility_glow(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.rand(2, 8, 32, 32)
        # ----------------------------------------------------------------------
        # Prepare the layer
        # ----------------------------------------------------------------------
        coupling = GlowCoupling(8, 16,
                                checkerboard_mask=False, invert_mask=False)
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = coupling(data)
        back = coupling.reverse(out)
        error_reco = torch.mean(torch.abs(back - data)).item()
        # -------------
        self.assertEqual(error_reco, 0)
        self.assertEqual(log_det.sum().item(), 0)
        # ----------------------------------------------------------------------
        # Randomise the weights of the layer
        # ----------------------------------------------------------------------
        init.kaiming_normal_(coupling.nn.conv_out.weight)
        init.ones_(coupling.nn.conv_out.bias)
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = coupling(data)
        back = coupling.reverse(out)
        error_reco = torch.mean(torch.abs(back - data)).item()
        # -------------
        self.assertLessEqual(error_reco, 1e-7)


if __name__ == '__main__':
    unittest.main()
