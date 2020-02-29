import os
import json
import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision
from tqdm import tqdm
from time import sleep
from data_io import ImageDataset
from .losses import NLLFlowLoss


class Trainer:
    """
    Performs automated training on a normalizing flow model

    :param model: The torch.nn Module to train
    :param data_path: the path to the root folder containing the database
    :param batch_size: the number of samples in each batch
    :param learning_rate: the learning rate for SGD
    :param saving_directory: where to save the results (network weights and
        image samples)
    :param device: either cpu or a gpu, the device on which training is to
        be performed
    :param weight_norm: whether to use weight norm for training (default:
        {False})
    :param data_augmentation: global probability of occurrence for the data
        augmentation (default:{0})
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self,
                 model: nn.Module,
                 data_path: str,
                 batch_size: int,
                 learning_rate: float,
                 saving_directory: str,
                 device: torch.device,
                 weight_norm: bool = False,
                 data_augmentation: float = 0.):
        self.__model = model
        self.__optimizer = \
            torch.optim.Adam(self.__model.parameters(), learning_rate)
        self.__loss = NLLFlowLoss(sigma=1.0, quantization=256, 
                                  bits_per_dim=True)
        self.__to_stop = False
        self.__saving_dir = saving_directory
        self.__training_dir = data_path
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__weight_norm = weight_norm
        self.__data_augmentation = data_augmentation
        self.__device = device
        self.__best_val = None
        # Prepare output dir
        print('---------------------------------------------------------------')
        print('The trained model will be saved at ' + self.__saving_dir)
        print('---------------------------------------------------------------')
        if not os.path.exists(self.__saving_dir):
            os.makedirs(self.__saving_dir)
        # Prepare the training logs
        self.__log_file = os.path.join(self.__saving_dir, "training.log")
        with open(self.__log_file, 'w') as file:
            file.write("epoch, train, val\n")

    # --------------------------------------------------------------------------
    # Main Processing Call
    # --------------------------------------------------------------------------

    def run(self):
        """
        Runs the trainer on the assigned model
        """
        self.__to_stop = False
        print('Running on ' + str(self.__device.type) + ' device')
        epoch = 1
        # ----------------------------------------------------------------------
        # Prepare Data Loaders
        # ----------------------------------------------------------------------
        train_loader = td.DataLoader(
            ImageDataset(self.__training_dir,
                         train=True,
                         data_augmentation=self.__data_augmentation),
            shuffle=True,
            batch_size=self.__batch_size,
            num_workers=4)
        val_loader = td.DataLoader(
            ImageDataset(self.__training_dir,
                         train=False,
                         data_augmentation=0),
            shuffle=True,
            batch_size=self.__batch_size * 2,
            num_workers=4)
        # ----------------------------------------------------------------------
        # Save the network configuration
        # ----------------------------------------------------------------------
        config = self.__model.config
        config["image_size"] = train_loader.dataset.image_size
        config["training_dir"] = self.__training_dir
        config["learning_rate"] = self.__learning_rate
        config["batch_size"] = self.__batch_size
        config["weight_norm"] = self.__weight_norm
        config["data_augmentation"] = self.__data_augmentation
        with open(os.path.join(self.__saving_dir, "config.json"), 'w') as file:
            json.dump(config, file)
        print("Model being trained is {0}".format(config["name"]))
        print("Data is at {0} and is of size {1}".format(config["training_dir"], 
            config["image_size"]))
        print("Batch size = {0} and learning rate = {1}".format(
            config["batch_size"], config["learning_rate"]))
        print("Weight Normalization is set to {}".format(config["weight_norm"]))
        print("Data Augmentation proportion is {}".format(
            config["data_augmentation"]))
        # ----------------------------------------------------------------------
        # Training Start
        # ----------------------------------------------------------------------
        while not self.__to_stop:
            # ------------------------------------------------------------------
            # Epoch start
            # ------------------------------------------------------------------
            print('----- Epoch {0} -----'.format(epoch))
            sleep(0.01)
            # ------------------------------------------------------------------
            # Training phase
            # ------------------------------------------------------------------
            if self.__weight_norm:
                self.__model.apply_weight_norm()
            running_loss = 0
            with tqdm(desc="Loss 0", total=len(train_loader)) as pbar:
                for batch_index, data in enumerate(train_loader):
                    loss = self._train(data.to(self.__device))
                    running_loss += loss
                    pbar.set_description(
                        "Loss {0:.5f}".format(running_loss / (batch_index + 1)))
                    pbar.update(1)
            training_loss = running_loss / len(train_loader)
            sleep(0.01)
            print("  |")
            print("  |--> Training loss : {0:.4f}".format(training_loss))
            print("  |")
            # ------------------------------------------------------------------
            # Validation phase
            # ------------------------------------------------------------------
            if self.__weight_norm:
                self.__model.remove_weight_norm()
            running_val_loss = 0
            with tqdm(desc="Val Loss 0", total=len(val_loader)) as pbar:
                for batch_index, data in enumerate(val_loader):
                    with torch.no_grad():
                        val_loss = self._evaluate(data.to(self.__device))
                    running_val_loss += val_loss
                    pbar.set_description("Val Loss {0:.5f}".format(
                        running_val_loss / (batch_index + 1)))
                    pbar.update(1)
            validation_loss = running_val_loss / len(val_loader)
            # ------------------------------------------------------------------
            sleep(0.01)
            print("  |")
            print("  |--> Validation loss : {0:.4f}".format(validation_loss))
            print("  |")
            # ------------------------------------------------------------------
            # Epoch end checkpoint
            # ------------------------------------------------------------------
            print('--..--**--..--**--..--**--..--**--..--**--..--**--..--**--')
            # Save the model weights only if we improved things
            if self.__best_val is None or \
                    validation_loss < 0.999 * self.__best_val:
                filename = os.path.join(
                    self.__saving_dir,
                    self.__model.config["name"] + ".pt".format(epoch))
                torch.save(self.__model.state_dict(), filename)
                self.__best_val = validation_loss
                print('    |')
                print('    |------> Saving model weights')
                print('    |')
            # Sample from the model and write the logs
            shape = (8, *data.size()[1:])
            self._sample(shape, os.path.join(self.__saving_dir,
                         self.__model.config["name"] + "_{0}.jpg".format(epoch)))
            self._write_log({"epoch": epoch,
                             "loss": training_loss,
                             "val_loss": validation_loss})

            epoch += 1
        # ----------------------------------------------------------------------
        # Training End
        # ----------------------------------------------------------------------
        print('Training Done !')

    # --------------------------------------------------------------------------
    # Sub Tasks
    # --------------------------------------------------------------------------

    def _train(self, data: torch.Tensor):
        """
        Performs one step of training
        """
        self.__model.train()
        self.__optimizer.zero_grad()
        output, sum_log_det = self.__model(data)
        loss = self.__loss(output, sum_log_det)
        loss.backward()
        self.__optimizer.step()
        return loss.item()

    def _evaluate(self, data: torch.Tensor):
        """
        Performs one step of validation
        """
        self.__model.eval()
        output, sum_log_det = self.__model(data)
        loss = self.__loss(output, sum_log_det)
        return loss.item()

    # --------------------------------------------------------------------------
    # Monitoring calls
    # --------------------------------------------------------------------------

    def _sample(self, shape: torch.Size, filename: str):
        """
        Performs inference on the network to get several samples

        :param shape: The input tensor size.
        :param filename: name of the output file
        """
        self.__model.eval()
        with torch.no_grad():
            output = self.__model.reverse(0.5 *
                torch.randn(shape).to(self.__device))
        # Write the output
        torchvision.utils.save_image(output, filename, nrow=2)

    def _write_log(self, data: dict):
        with open(self.__log_file, 'a') as file:
            file.write(
                "{0}, {1:.5f}, {2:.5f}\n".format(data["epoch"],
                                                 data["loss"],
                                                 data["val_loss"]))
