import unittest
import torch.nn.init as init
from models import RealNVP, Glow
from utils.network_tools import *


class WeightSaveLoadTest(unittest.TestCase):
    
    def test_real_nvp_save_load(self):
        # ----------------------------------------------------------------------
        # Data preparation and Run through dummy network
        # ----------------------------------------------------------------------
        data = torch.rand(2, 3, 32, 32)
        cwd = os.path.dirname(os.path.realpath(__file__))
        net = RealNVP(context_blocks=2, input_channels=3,
                           hidden_channels=64, quantization=65536)
        output, _ = net(data)
        # ----------------------------------------------------------------------
        # Save and load back model
        # ----------------------------------------------------------------------
        # Save the model configuration
        with open(os.path.join(cwd, "config.json"), 'w') as file:
            json.dump(net.config, file)
        # Save the model state dictionary
        filename = os.path.join(cwd, net.config["name"] + ".pt")
        torch.save(net.state_dict(), filename)
        # Load it back
        loaded_model, config = load_model(cwd)
        # ----------------------------------------------------------------------
        # Assert the output is as expected
        # ----------------------------------------------------------------------
        new_output, _ = loaded_model(data)
        error = torch.mean(torch.abs(new_output - output)).item()
        self.assertLessEqual(error, 5e-5)
        self.assertEqual(config, net.config)

    def test_glow_save_load(self):
        # ----------------------------------------------------------------------
        # Data preparation and Run through dummy network
        # ----------------------------------------------------------------------
        data = torch.rand(2, 3, 32, 32)
        cwd = os.path.dirname(os.path.realpath(__file__))
        net = Glow(context_blocks=3, flow_steps=3, input_channels=3,
                        hidden_channels=32, quantization=65536)
        output, _ = net(data)
        # ----------------------------------------------------------------------
        # Save and load back model
        # ----------------------------------------------------------------------
        # Save the model configuration
        with open(os.path.join(cwd, "config.json"), 'w') as file:
            json.dump(net.config, file)
        # Save the model state dictionary
        filename = os.path.join(cwd, net.config["name"] + ".pt")
        torch.save(net.state_dict(), filename)
        # Load it back
        loaded_model, config = load_model(cwd)
        # ----------------------------------------------------------------------
        # Assert the output is as expected
        # ----------------------------------------------------------------------
        new_output, _ = loaded_model(data)
        error = torch.mean(torch.abs(new_output - output)).item()
        self.assertLessEqual(error, 5e-5)
        self.assertEqual(config, net.config)


if __name__ == '__main__':
    unittest.main()
