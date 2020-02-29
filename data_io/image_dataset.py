import os
import torch
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(torch.utils.data.Dataset):
    """
    A Simple dataset that loads images from a folder without any particular 
    structure. Should be given the path to a root folder containing 'train' 
    and 'val' sub-folders.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, dataset_path: str,
                 train: bool = True,
                 data_augmentation: float = 0.25):
        """
        :param dataset_path: path towards the dataset folder
        :param train: whether to create dataset from training set or val set
        :param data_augmentation: global probability of occurrence for the data
         augmentation
        """
        super(ImageDataset, self).__init__()
        if train:
            data_path = os.path.join(dataset_path, 'train')
        else:
            data_path = os.path.join(dataset_path, 'val')
        # Get the paths to all images
        self.__paths = [os.path.join(data_path, f)
                        for f in os.listdir(data_path) if not f.startswith('.')]
        # Prepare data augmentation transforms
        prob = data_augmentation / 2
        self.__transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.7,
                                       contrast=0.7,
                                       saturation=0.7,
                                       hue=0.2)],
                                   p=prob),
            transforms.RandomHorizontalFlip(p=prob)
        ])

    # --------------------------------------------------------------------------
    # Get a sample from the dataset
    # --------------------------------------------------------------------------

    def __getitem__(self, idx: int):
        """
        Operator overloading to be able to call ImageDataset[idx]
        :param idx: which sample to return
        :return: a torch tensor corresponding to an image (3 x H x W)
        """
        # Read image
        image = Image.open(self.__paths[idx])
        # Apply degradations
        image = self.__transforms(image)
        # Return torch tensor
        return transforms.functional.to_tensor(image)

    # --------------------------------------------------------------------------
    # Length of the dataset
    # --------------------------------------------------------------------------

    def __len__(self):
        """
        Operator overloading to be able to call len(ImageDataset object)
        :return: int, the number of samples available in the whole dataset
        """
        return int(len(self.__paths))

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def image_size(self):
        image = Image.open(self.__paths[0])
        image = transforms.functional.to_tensor(image)
        return image.shape
