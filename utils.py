import numpy as np
from scipy import stats
import torch

from torch import nn
from torch import optim

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

NUM_CLASSES = 20

def get_mean_std(data_dir):
    train_data = datasets.ImageFolder(data_dir,transform=transforms.ToTensor())
    train_idx, test_idx = split_data(train_data,0.2)
    train_sampler = SubsetRandomSampler(train_idx)
    loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=10)
    nimages = 0
    mean = 0.0
    var = 0.0
    for batch, _ in loader:
        #print(batch.shape)
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)
        if nimages%500 == 0:
            print(mean)
            print(var)
            print(nimages)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print(mean)
    print(std)
    
    return mean, std
    
def split_data(train_data, valid_size):
    num_train_data = len(train_data)
    indices = list(range(num_train_data))

    split = int(np.floor(valid_size*num_train_data/NUM_CLASSES))
    total_per_class = int(np.floor(num_train_data/NUM_CLASSES))
    # np.random.shuffle(indices)

    train_idx = indices[split:total_per_class:2]
    test_idx = indices[:split:2]
    
    for itr in range(NUM_CLASSES-1):
        train_idx.extend(indices[((itr+1)*total_per_class) + split:(itr+2)*total_per_class:2])
        test_idx.extend(indices[(itr+1)*total_per_class:split + (itr+1)*total_per_class:2])
        print(((itr+1)*total_per_class) + split,(itr+2)*total_per_class)
        print((itr+1)*total_per_class,split + (itr+1)*total_per_class)
    print('Train index length',len(train_idx))
    print('Test index length',len(test_idx))
    
    return train_idx, test_idx
    
## Function to Split original data into train and test followed by loading of the data
def load_split_test_train(data_dir,valid_size = 0.2):
    #mean, std = get_mean_std(data_dir)
    mean = torch.tensor([0.4865, 0.3409, 0.3284])
    std = torch.tensor([0.1940, 0.1807, 0.1721])
    train_transforms = transforms.Compose([transforms.Resize((512,288)),transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean,std)])
    test_transforms = transforms.Compose([transforms.Resize((512,288)),transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(mean,std)])

    data_transforms_A = transforms.Compose([transforms.RandomHorizontalFlip(1.0),transforms.ToTensor(), transforms.Normalize(mean,std)])
    data_transforms_B = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    data_transforms_C = transforms.Compose([transforms.RandomCrop(size=(288,512),padding=20, pad_if_needed=True, fill=0, padding_mode='symmetric'), transforms.ToTensor(), transforms.Normalize(mean,std)])
    train_data = torch.utils.data.ConcatDataset([datasets.ImageFolder(data_dir,transform=data_transforms_A), datasets.ImageFolder(data_dir, transform=data_transforms_B), datasets.ImageFolder(data_dir, transform=data_transforms_C)])
       
    train_idx, test_idx = split_data(train_data,0.2)
    
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
    test_loader = torch.utils.data.DataLoader(train_data,sampler=test_sampler, batch_size=64)

    return train_loader, test_loader