import unittest
import torch
from layers import Squeeze2x2, AlternateSqueeze2x2


class SqueezeTest(unittest.TestCase):

    def test_invertibility_squeeze2x2(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.rand(2, 3, 32, 32)
        # ----------------------------------------------------------------------
        # Prepare the layer
        # ----------------------------------------------------------------------
        squeeze = Squeeze2x2()
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = squeeze(data)
        back = squeeze.reverse(out)
        error_reco = torch.mean(torch.abs(back - data)).item()
        # -------------
        self.assertEqual(error_reco, 0)
        self.assertEqual(log_det.item(), 0)
        
    def test_behaviour_squeeze2x2(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).view(1, 1, 2, 4)
        target = torch.Tensor([[1, 3], [2, 4], [5, 7], [6, 8]]).view(1, 4, 1, 2)
        # ----------------------------------------------------------------------
        # Prepare the layer
        # ----------------------------------------------------------------------
        squeeze = Squeeze2x2()
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, _ = squeeze(data)
        error_reco = torch.mean(torch.abs(target - out)).item()
        # -------------
        self.assertEqual(error_reco, 0)

    def test_invertibility_alternate_squeeze2x2(self):
        # ----------------------------------------------------------------------
        # Prepare some dummy data
        # ----------------------------------------------------------------------
        data = torch.rand(2, 4, 32, 32)
        # ----------------------------------------------------------------------
        # Prepare the layer
        # ----------------------------------------------------------------------
        squeeze = AlternateSqueeze2x2()
        # ----------------------------------------------------------------------
        # Assess the results are as expected
        # ----------------------------------------------------------------------
        out, log_det = squeeze(data)
        back = squeeze.reverse(out)
        error_reco = torch.mean(torch.abs(back - data)).item()
        # -------------
        self.assertEqual(error_reco, 0)
        self.assertEqual(log_det.item(), 0)


if __name__ == '__main__':
    unittest.main()
