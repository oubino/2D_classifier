# custom dataset class for the images and masks

import torch
from torch.utils.data import Dataset#, DataLoader
import os
import numpy as np

# CTDataset
class HeartDataset(Dataset):
    """Heart dataset"""

    def __init__(self, root, transform_train =None, transform_test = None, test = False, train = False):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Train Data")))) # ensure they're aligned & index them
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.test = False
        self.train = False
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx): # convert tensor to list to index items
            idx = idx.tolist() 
            
        img_path = os.path.join(self.root, "Train Data", self.imgs[idx]) # image path is combination of root and index 
        img = np.load(img_path) # image read in as numpy array
        
        sample = {'image': img} # both are nd.arrays, stored in sample dataset
        sample['idx'] = idx # should print out which image is problematic
        sample['patient'] = self.imgs[idx]

        # load in structure coords
        # load in beat label
        label_path = os.path.join(self.root, "OHE_trainval.npy") # image path is combination of root and index 
        labels = np.load(label_path)
        beat_label = labels[idx]
        if beat_label[0] == 0: # normal
            new_label = 0
        elif beat_label[0] == 1:
            new_label = 1
        sample['label'] = new_label
        
        if (self.transform_train) and (self.test == False):
            sample = self.transform_train(sample)
        if (self.transform_test) and (self.test == True):
            sample = self.transform_test(sample)
        
        return sample
    
    
    def __len__(self):
        return len(self.imgs) # get size of dataset
    
    def __pat__from__index(self, idx):
        return self.imgs[idx]

    def __test__(self):
      self.test = True
      self.train = False
    
    def __train__(self):
      self.train = True
      self.test = False

