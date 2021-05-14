# dataloaders

from torchvision import transforms #, datasets, models
import torch
from torch.utils.data import DataLoader

from data_loading import dataset_class as D
from data_loading import transforms as T
import settings as S
import numpy as np

trans_plain = transforms.Compose([T.Normalise(), T.ToTensor()])
#trans_augment = transforms.Compose([T.ToTensor()])
trans_augment = transforms.Compose([T.Normalise(), T.HorizontalFlip(), T.Rotation(), T.Shifting(), T.RandomErasing(), T.Noise(), T.ToTensor()])
dataset = D.HeartDataset(S.root, transform_train = trans_augment, transform_test = trans_plain, test = False)

def init(fold, train_ids, test_ids):
    # initialise dataloader 
    # split train_ids into val and train
    np.random.shuffle(train_ids)
    index = int(len(train_ids)/10) # val ids are first 10 percent  
    val_ids = train_ids[:index]
    train_ids = train_ids[index:]
    
    global test_set_ids, val_set_ids
    test_set_ids = []
    val_set_ids = []
    for i in test_ids:
        test_set_ids.append(dataset._HeartDataset__pat__from__index(i))
    for i in val_ids:
        val_set_ids.append(dataset._HeartDataset__pat__from__index(i))
    
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    

    global dataloaders

    dataloaders = {
    'train': DataLoader(dataset, batch_size=S.batch_size, sampler= train_subsampler),
    'test': DataLoader(dataset, batch_size=S.batch_size_test, sampler= test_subsampler),
    'val': DataLoader(dataset, batch_size=S.batch_size, sampler= val_subsampler)  
    }

