import numpy as np
import torch

from torch import nn
from torch import optim

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import xlsxwriter


data_dir = 'test_frame_data'

### This class is taken from the gist shared by Andrew Jong (gist.github.com/andrewjong)
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

## Function to load original data loading of the data
def load_all_data(data_dir):
    test_data = ImageFolderWithPaths(data_dir,transform=transforms.ToTensor())

    num_train_data = len(test_data)
    indices = list(range(num_train_data))

    skip_ind = indices[0::10]
  
    sub_data = torch.utils.data.Subset(test_data,skip_ind)
    test_loader = torch.utils.data.DataLoader(sub_data,batch_size=1)

    return test_loader

testloader = load_all_data(data_dir)
pretrained_weights = torch.load('trained_model.pth')

model = models.resnet18(pretrained=False)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 20)
model.load_state_dict(pretrained_weights)

xls = []
workbook = xlsxwriter.Workbook('features512_ALLlayers_resnet18.xlsx')
worksheet = workbook.add_worksheet()
worksheet2 = workbook.add_worksheet()

#print(torch.nn.Sequential(*list(model.children())[:-5],torch.nn.AdaptiveAvgPool2d(1)))
### strip the last layer
#feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1],torch.nn.AdaptiveAvgPool2d(1))

row= 0
for inputs, labels, paths in testloader:
    input, label, path = inputs,labels,paths
	#following function extracts features for input
    output = feature_extractor(input)
    np_out = output.flatten().detach().numpy()
 
    if row % 10 == 0:
	    print(row)
    worksheet2.write_column(row,0,path)
    col = 0
    for i in range(len(np_out)):
        worksheet.write_number(row, col, np_out[i])
        col = col + 1
    row = row + 1

workbook.close()