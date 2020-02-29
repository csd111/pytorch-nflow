import os
import torch
import json
from models import ModelLoader

# ------------------------------------------------------------------------------
# Weight Handling
# ------------------------------------------------------------------------------


def load_model(network_path: str):
    """
    Utility function to reload a pre-trained network

    :param network_path: path to the directory
    :return the pre-trained torch.nn.Module + the configuration of the training
    """
    # Load the config file of the network
    with open(os.path.join(network_path, 'config.json'), 'r') as file:
        config = json.load(file)
    # Instantiate the corresponding network
    model_class = ModelLoader.get_model(config["name"])
    model = model_class.from_config(config)
    # Get the weights file
    weights_path = os.path.join(network_path, config["name"] + ".pt")
    # Load the weights
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model, config


def transfer_model_weights(model: torch.nn.Module,
                           trained_model: torch.nn.Module):
    """
    Used to transfer the weights of a pre-trained model into a network with a
    (potentially) different architecture. Looks for layers with the same name
    and tries to load the corresponding part of the pre-trained state-dict

    :param model: the model you want to train (the destination)
    :param trained_model: the pre-trained model (the source)

    """
    trained_params = trained_model.state_dict()
    current_params = model.state_dict()
    incompatible_count = 0
    # For each learnable parameter tensor in the source network
    for name1, param1 in trained_params.items():
        if name1 in current_params:
            # Transfer the weights if the tensors are compatible
            if current_params[name1].data.size() == param1.data.size():
                current_params[name1].data.copy_(param1.data)
            else:
                incompatible_count += 1
        else:
            incompatible_count += 1
    model.load_state_dict(current_params, strict=False)
    print("Found {0} incompatible layers".format(str(incompatible_count)))
    return model

# ------------------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------------------


def count_parameters(model: torch.nn.Module):
    """
    Utility function to count the number of trainable parameters in a model

    :param model: the neural network model as a torch.Module
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_layers(model: torch.nn.Module):
    """
    Utility function to count the number of trainable layers in a model. Please
    note this function considers a layer any tensor-represented group of
    trainable parameters. This means that, for example, it splits special layers
    like GRUs into multiple ones when counting.

    :param model: the neural network model as a torch.Module
    """
    return sum(1 for p in model.parameters() if
               (p.requires_grad and p.numel() > 1))
