import os
import torch
import json
from models import RealNVP, Glow


def load_model(network_path: str):
    """
    Utility function to reload a pre-trained network
    :param network_path: path to the directory
    :return the pre-trained torch.nn.Module + the image size used for training
    """
    # Load the config file of the network
    with open(os.path.join(network_path, 'config.json'), 'r') as file:
        config = json.load(file)
    # Instantiate the corresponding network
    if config["name"] == "real-nvp":
        model = RealNVP(context_blocks=int(config["num_levels"]),
                        input_channels=3,
                        hidden_channels=int(config["num_features"],
                        full_graph=config["full_graph"]))
    elif config["name"] == "glow":
        model = Glow(context_blocks=int(config["num_levels"]),
                     flow_steps=int(config["num_flows"]),
                     input_channels=3,
                     hidden_channels=int(config["num_features"],
                     full_graph=config["full_graph"],
                     lu_decomposition=config["lu_decomposition"]))
    else:
        raise Exception('Unknown normalizing flow module')
    # Get the weights file
    weights_path = os.path.join(network_path, config["name"] + ".pt")
    # Load the weights
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model, tuple(config["image_size"])
