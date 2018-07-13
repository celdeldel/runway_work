import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

data_transforms = transforms.Compose([transforms.RandomSizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#data_dir = '/Users/celdel/workbitch/work_pytorch/'
image_datasets = {x: datasets.ImageFolder(x,data_transforms) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=4)for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

plotly.tools.set_credentials_file(username='celdeldel', api_key='8mQ0wRl85oCkuarGxh2d')


# finetuning
model_ft = models.resnet34(pretrained=True)



num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

layers = [model_ft.conv1.parameters(), model_ft.bn1.parameters(), model_ft.relu.parameters(), model_ft.maxpool.parameters(), model_ft.layer1.parameters(), model_ft.layer2.parameters(), model_ft.layer3.parameters(), model_ft.layer4.parameters(), model_ft.avgpool.parameters(),model_ft.fc.parameters()]
lrs = [0.000001,0.000001,0.000001,0.000001,0.00001,0.00001,0.00001,0.0001,0.001,0.01]


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([{'params': p, 'lr': l} for p,l in zip(layers, lrs)], momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0, last_epoch=-1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    #since = time.time()
    loss_train = []
    loss_test = []
    acc_test = []
    acc_train = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'train':
                loss_train.append(epoch_loss)
                acc_train.append(epoch_acc)
            else:
                loss_test.append(epoch_loss)
                acc_test.append(epoch_acc)

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    #time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model, loss_train, loss_test, acc_train, acc_test



model, loss_train, loss_test, acc_train, acc_test= train_model(model_ft, criterion, optimizer, scheduler, num_epochs=10)

trace0 = go.Scatter(
    x=[0,1, 2, 3, 4],
    y = loss_train
)
trace1 = go.Scatter(
    x=[0,1, 2, 3, 4],
    y = loss_test
)

trace2 = go.Scatter(
    x=[0,1, 2, 3, 4],
    y = acc_train
)
trace3 = go.Scatter(
    x=[0,1, 2, 3, 4],
    y = acc_test
)


data = [trace0, trace1, trace2, trace3]

py.plot(data, filename = 'resnet34 as features extractor', auto_open=True)
