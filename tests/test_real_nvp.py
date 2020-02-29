import unittest
import torch
from models import RealNVP
from training.losses import NLLFlowLoss


class RealNVPTest(unittest.TestCase):

    def test_invertibility_real_nvp(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.rand(2, 8, 32, 32)
        # ----------------------------------------------------------------------
        # Prepare the layer with default init
        # ----------------------------------------------------------------------
        coupling = RealNVP(context_blocks=3, input_channels=8,
                           hidden_channels=32, quantization=65536)
        coupling.eval()
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = coupling(data)
        back = coupling.reverse(out)
        error_reco = torch.mean(torch.abs(back - data)).item()
        # -------------
        self.assertLessEqual(error_reco, 1e-5)
        self.assertNotEqual(log_det.sum().item(), 0)
        # ----------------------------------------------------------------------
        # Apply and remove weight norm and check the results don't change
        # ----------------------------------------------------------------------
        coupling.apply_weight_norm()
        out2, log_det2 = coupling(data)
        coupling.remove_weight_norm()
        back2 = coupling.reverse(out)
        error_out = torch.mean(torch.abs(out2 - out)).item()
        error_back = torch.mean(torch.abs(back2 - back)).item()
        # -------------
        self.assertLessEqual(error_out, 5e-5)
        self.assertLessEqual(error_back, 1e-8)

    def test_training_real_nvp(self):
        # ----------------------------------------------------------------------
        # Prepare the layer with default init
        # ----------------------------------------------------------------------
        coupling = RealNVP(context_blocks=2, input_channels=3,
                           hidden_channels=64, quantization=65536)
        # ----------------------------------------------------------------------
        # Train for a couple of batches
        # ----------------------------------------------------------------------
        optimizer = torch.optim.Adam(coupling.parameters(), 0.0001)
        loss = NLLFlowLoss(sigma=1.0, quantization=65536, bits_per_dim=True)
        for _ in range(20):
            optimizer.zero_grad()
            data = torch.rand(2, 3, 32, 32)
            out, log_det = coupling(data)
            nll = loss(out, log_det)
            nll.backward()
            optimizer.step()
        # ----------------------------------------------------------------------
        # Assess the network is still invertible
        # ----------------------------------------------------------------------
        coupling.eval()
        data = torch.rand(2, 3, 32, 32)
        out, log_det = coupling(data)
        back = coupling.reverse(out)
        error_reco = torch.mean(torch.abs(back - data)).item()
        # -------------
        self.assertLessEqual(error_reco, 1e-5)
        self.assertNotEqual(log_det.sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
