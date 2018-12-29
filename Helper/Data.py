from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import collections
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as transform

#Transformations
Norm = transform.Compose([transform.Resize((224,224)),transform.ToTensor(), transform.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
To_tensor = transform.ToTensor()
Resize = transform.Resize((224,224))
To_pil = transform.ToPILImage()



class Whales(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.

        """
        self.Whales_images = pd.read_csv(csv_file)
        self.root_dir = root_dir


    def __len__(self):
        return len(self.Whales_images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.Whales_images.iloc[idx, 0])
        image = Image.open(img_name)

        image = Norm(image)
        whale = self.Whales_images.iloc[idx, 1:].values
        whale = whale.astype('float').reshape(-1, 1)
        sample = {'image': image, 'landmarks': whale}


        return sample
Whales_dataset = Whales(csv_file='Data/NEW_Train.csv',
                                    root_dir='Data/Images/Augmentation/')
dataloader = DataLoader(Whales_dataset, batch_size=32,
                        shuffle=True)
Split = len(Whales_dataset)

Cross_validation, _ = torch.utils.data.random_split(Whales_dataset, (2000, Split - 2000))
dataloader_CV = DataLoader(Cross_validation,batch_size=32,
                        shuffle=True)
