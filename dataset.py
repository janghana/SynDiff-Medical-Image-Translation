from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch

class MRIDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.images = []

        # Sort patient directories to maintain consistent ordering
        patient_dirs = sorted(os.listdir(directory))

        for patient_dir in patient_dirs:
            patient_dir_path = os.path.join(directory, patient_dir)
            
            # Sort images to maintain consistent ordering
            image_files = sorted(os.listdir(patient_dir_path))
            
            for image_file in image_files:
                if image_file.endswith(".png"):
                    image_path = os.path.join(patient_dir_path, image_file)
                    image = Image.open(image_path).convert("L")
                    self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

class CTDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.images = []

        # Sort patient directories to maintain consistent ordering
        patient_dirs = sorted(os.listdir(directory))

        for patient_dir in patient_dirs:
            patient_dir_path = os.path.join(directory, patient_dir)
            
            # Sort images to maintain consistent ordering
            image_files = sorted(os.listdir(patient_dir_path))
            
            for image_file in image_files:
                if image_file.endswith(".png"):
                    image_path = os.path.join(patient_dir_path, image_file)
                    image = Image.open(image_path).convert("L") # to Gray Scale
                    self.images.append(image)  # No cropping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

def CreateDatasetSynthesis(phase, input_path, contrast1 = 'CT', contrast2 = 'MRT1'):
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
        transforms.Resize((256, 256)),  # Resize image to 256x256
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])
    
    ct_target_file = input_path + "/CT/"
    data_fs_ct = CTDataset(ct_target_file, transform=transform)

    mrt1_target_file = input_path + "/MRT1/"
    data_fs_mrt1 = MRIDataset(mrt1_target_file, transform=transform)

    data_fs_ct = torch.stack([data_fs_ct[i] for i in range(len(data_fs_ct))])
    data_fs_mrt1 = torch.stack([data_fs_mrt1[i] for i in range(len(data_fs_mrt1))])
    
    dataset=torch.utils.data.TensorDataset(data_fs_ct, data_fs_mrt1)  

    return dataset
