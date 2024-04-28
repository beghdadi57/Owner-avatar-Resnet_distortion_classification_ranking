import numpy as np
from scipy import stats
import torch

from torch import nn
from torch import optim

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils import load_split_test_train
from .plot import plot_results

data_dir = 'frame_data'
NUM_CLASSES = 20
#mean = tensor([0.4865, 0.3409, 0.3284])
#std = tensor([0.1940, 0.1807, 0.1721])


## Call split and load function
trainloader, testloader = load_split_test_train(data_dir, .2)
#print(trainloader.dataset.classes, len(trainloader), len(testloader))
print(len(trainloader), len(testloader))
im, lab = iter(trainloader).next()
print('Labels:', lab)

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
model = models.resnet18(pretrained=True)
#model = models.resnet101(pretrained=False)
#model = models.vgg16(pretrained=False)
#model = torch.load('livemodel_32batch_evry15_resnet18Adam_200ep_lr_pt001.pth')

num_ftrs = model.fc.in_features 	#for ResNet
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)#, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2,verbose=False)
model.to(device)


epochs = 50
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
train_acc, test_acc = [], []
srocc = []
total_train = 0
correct_train = 0
total_test = 0
correct_test = 0
test_loss = 0


for epoch in range(epochs):
    train_accuracy = 0
    print("Epoch ",epoch)
    print("Training... ")
    steps = 0
    model.train()
    scores_arr = []
    pred_arr = []
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outs = model.forward(inputs)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      
        _, predicted = torch.max(outs.data, 1)
        total_train += labels.nelement()
        correct_train += predicted.eq(labels.data).sum().item()
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(running_loss/len(trainloader))
    train_acc.append(train_accuracy)
        #if steps % print_every == 0:
   
    test_accuracy = 0
    model.eval()
    print(steps,"\nValidation... ")
    steps = 0
    for inputs, labels in testloader:
        steps += 1
        inputs, labels = inputs.to(device),labels.to(device)


        outs = model.forward(inputs)
        batch_loss = criterion(outs, labels)
        test_loss += batch_loss.item()

        _, pred = torch.max(outs.data, 1)
        total_test += labels.nelement()
        correct_test += pred.eq(labels.data).sum().item()
        
        scores_arr.extend(labels.data.cpu())
        pred_arr.extend(pred.cpu())
    scheduler.step(test_loss)
    test_accuracy = 100 * correct_test / total_test
    test_losses.append(test_loss/len(testloader))
    test_acc.append(test_accuracy)
    srocc_iter, pv = stats.spearmanr(np.array(pred_arr),np.array(scores_arr))
    srocc.append(srocc_iter)
        
    print(steps,"\nTrain accuracy:" + str(train_accuracy))
    print("Train loss:" + str(running_loss/len(trainloader)))
    print("Test loss:" + str(test_loss/len(testloader)))
    print("Test accuracy:"+ str(test_accuracy))
    print("Test SROCC:"+ str(srocc_iter))

    running_loss = 0
    test_loss = 0
    total_train = 0
    correct_train = 0
    total_test = 0
    correct_test = 0
    med_srocc = np.median(srocc)
    print("Median SROCC:"+str(med_srocc))
    mean_srocc = np.mean(srocc)
    print("Mean SROCC:"+str(mean_srocc))

med_srocc = np.median(srocc)
print("Median SROCC:"+str(med_srocc))
mean_srocc = np.mean(srocc)
print("Mean SROCC:"+str(mean_srocc))

torch.save(model.state_dict(), 'trained_model.pth')

plot_results(train_losses,train_acc,test_losses,test_acc)