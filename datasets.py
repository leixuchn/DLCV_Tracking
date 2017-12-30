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
        ## No transformation done
        #return sample = {'previmg': prev_img,
        ##        'currimg': curr_img,
        ##        'currbb' : currbb
        ##          }
        previmg_data = self.bboxes[idx]
        prev_image_name = previmg_data[0]
        prev_image_trackId = previmg_data[6]
        prev_vv = previmg_data[5]
        next_idx = idx + 1
        currimg_data = self.bboxes[next_idx]
        curr_image_name = currimg_data[0]
        curr_vv = currimg_data[5]
        while((prev_vv == curr_vv) and (prev_image_name == curr_image_name)):
            next_idx = next_idx + 1
            currimg_data = self.bboxes[next_idx]
            curr_image_name = currimg_data[0]
            curr_vv = currimg_data[5]

        curr_image_trackId = currimg_data[6]
        while((prev_vv == curr_vv) and (prev_image_trackId != curr_image_trackId)):
            next_idx = next_idx + 1
            currimg_data = self.bboxes[next_idx]
            curr_image_trackId = currimg_data[6]
            curr_vv = currimg_data[5]
        sample = {'previmg': prev_image_name, 'currimg': curr_image_name, 'currbb': self.getbb(next_idx)}
        return (sample)

    def getbb(self, idx):
        xmin = self.bboxes[idx][1]
        ymin = self.bboxes[idx][2]
        xmax = self.bboxes[idx][3]
        ymax = self.bboxes[idx][4]
        return ([xmin, ymin, xmax, ymax])









