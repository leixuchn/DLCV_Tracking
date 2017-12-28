from __future__ import print_function, division
import os
import torch
import numpy as np
#from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import make_labels_imnet_vedio as label_helper

import warnings
warnings.filterwarnings("ignore")

class DLCVDataset(Dataset):
    """DLCV Dataset"""

    def __init__(self, dataset_path, label_type, transform=None):
        self.dataset_path = dataset_path
        self.bboxes, self.imageNameList, self.totalImages = label_helper.main(dataset_path, label_type)


    def __len__(self):
        return self.totalImages

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        #TODO: Discuss in the team meeting before doing any transform
        #if (self.transform):
        #    sample = self.transform(sample)
        return sample

    def get_sample(self, idx):
        #TODO: again disuss what all is required in one sample?
        # At present it will return (image file path, bbox for that image)
        # bbox consists of x,y coordinates of annotations & few other parameters
        #return (self.imageNameList[idx], self.bboxes[idx])
        return (self.bboxes[idx])