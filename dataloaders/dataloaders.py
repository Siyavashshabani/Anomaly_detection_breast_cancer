import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch 
from torchvision.transforms import functional as F
# from torchvision.transforms import InterpolationMode
class RandomHorizontalFlipTensor(object):
    """Horizontally flip the given tensor randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            return tensor.flip(dims=[2])
        return tensor

class RandomRotateTensor(object):
    """Rotate the given tensor by a certain angle."""
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, tensor):
        angle = (torch.rand(1) * 2 - 1) * self.degrees  # Random angle between -degrees and +degrees
        # Rotate the tensor using BILINEAR interpolation from PIL
        return F.rotate(tensor, angle.item(), interpolation= Image.NEAREST)

# class RandomRotateTensor(object):
#     """Rotate the given tensor by a certain angle."""
#     def __init__(self, degrees):
#         self.degrees = degrees

#     def __call__(self, tensor):
#         angle = (torch.rand(1) * 2 - 1) * self.degrees  # Random angle between -degrees and +degrees
#         # Convert tensor to PIL image, rotate, then convert back to tensor
#         image = F.to_pil_image(tensor)
#         rotated_image = F.rotate(image, angle.item(), resample=Image.BILINEAR)
#         return F.to_tensor(rotated_image)

class RandomCropTensor(object):
    """Crop randomly the image in a sample."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, tensor):
        h, w = tensor.shape[1], tensor.shape[2]
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h, (1,)).item()
        left = torch.randint(0, w - new_w, (1,)).item()

        return tensor[:, top: top + new_h, left: left + new_w]


class RandomAffineTensor(object):
    """Random affine transformation of the image keeping center invariant"""
    def __init__(self, degrees, translate, scale):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, tensor):
        angle = (torch.rand(1) * 2 - 1) * self.degrees
        scale = torch.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]
        translate = [(torch.rand(1) * 2 - 1) * t for t in self.translate]
        return F.affine(tensor, angle=angle.item(), translate=translate, scale=scale.item(), shear=0, interpolation=None)

class AddGaussianNoise(object):
    """Add Gaussian noise to the image."""
    def __init__(self, mean=0.01, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Initialize dataset by loading paths to .npy files.

        Args:
            folder_path (str): Path to the directory containing .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.image_filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.transform = transform

    def __len__(self):
        """
        Return the total number of files in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, index):
        """
        Generate one sample of data.

        Args:
            index (int): Index of the file in the filenames list.

        Returns:
            tuple: (image, image) where both elements are the same numpy array loaded from a .npy file.
        """
        img_path = self.image_filenames[index]
        image = np.load(img_path)  # Load image data from .npy file
        if self.transform:
            image = self.transform(image)
        return image, image  # Return the same image as both input and target

def get_transform():
    """
    Define and return the necessary transformations.

    Returns:
        torchvision.transforms.Compose: Transformation pipeline.
    """
    return transforms.Compose([
        # Add any specific transformations you need, e.g., resizing, normalization
        transforms.ToTensor(),  # Convert arrays to PyTorch tensors
    ])

def create_data_loaders(train_folder, batch_size=32, num_workers=0, shuffle=True):
    """
    Create and return data loaders for training.

    Args:
        train_folder (str): Path to the training data folder.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): How many subprocesses to use for data loading.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset.
    """
    transform = get_transform_augment(augment=True)
    dataset = ImageFolderDataset(train_folder, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

# Example of how to use this module can be added here or in a separate file
# if __name__ == "__main__":
#     # Example usage: create a data loader for a specific directory
#     train_loader = create_data_loaders('./datasets/normal', batch_size=32, shuffle=True)


def get_transform_augment(augment=False):
    base_transforms = [
        # Assuming the data is already loaded as tensor, normalized and properly shaped
        transforms.ToTensor(),  # Convert arrays to PyTorch tensors

    ]

    if augment:
        # Data augmentation transforms specific to grayscale images
        augmentation_transforms = [
            RandomHorizontalFlipTensor(p=0.5),
            # AddGaussianNoise(),
            # RandomRotateTensor(degrees=15),
            # RandomCropTensor(output_size = 128),
            # RandomAffineTensor(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

        ]
        return transforms.Compose( base_transforms+ augmentation_transforms)
    else:
        return transforms.Compose(base_transforms)