#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 16:00:50 2018

@author: celdel



very simple net 

based on youtube tutorial
https://www.youtube.com/watch?v=oSirQZ_L7Q8&t=67s
https://www.youtube.com/watch?v=I_e---jO3mo

conclusion on this network

has been upgrading itself to this performances for the loss test function (4 avril) for 100000 iteration of the training loop
loss : -0.833591341972
loss test :-0.717231690884

9 avril : apres 100 000 iteratios de la training loop
loss : -0.8343653082847595
accuracy :0.5818965517241379
loss test :-0.7257752418518066

16 avril : apres 50000 iterations
loss : -0.8351393342018127
loss : -0.8351393342018127
accuracy :0.5818965517241379
loss test :-0.7251498699188232

24 mai :jai modifié la fonction loss a losscrossentropy
loss : 0.3944507837295532
accuracy :0.9022556390977443
loss test :0.5383819937705994

loss : 0.3944507837295532
accuracy :0.9022556390977443
loss test :0.5379353761672974

"""


import torch
from torch import autograd, nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import sklearn
import face_recognition
import undeuxtrois
import copy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import undeuxtrois

batch_size = 1292
input_size = 128
num_classes = 2
hidden_size = 20
learning_rate = 0.001
#my_file = "all_img_names.txt"
str_brand = "dries-van-noten"
number_of_epoch = 2000
name_file = "new_names.txt"

#loads the data if needed my_file is : 'all_img_names.txt' and str-brand example : 'dries-van-noten'
#e_c_n = undeuxtrois.www_l_c_encodings_from_file(my_file,str_brand)
e_c_n = enc = undeuxtrois.www_l_c_encodings_from_file(name_file,str_brand)[0]
input_brut = e_c_n[0] # on recupere les endocings
# penser a recuperer les labels pour plus tard aussi
input_labels = e_c_n[1]#on recupere les labels
#utiliser cross_validation pour separer les donnees en training et en test
#data_train_0, data_test_0, target_train_0, target_test = train_test_split(input_brut,input_labels,test_size=0.3, random_state=123)
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(input_brut,input_labels, test_size=0.20, random_state=123)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(input_brut,input_labels, test_size=0.26, random_state=123)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(input_brut,input_labels, test_size=0.33, random_state=123)

batch_size=len(X_train_0)



#transformer en pytorch tensor
data_train_copy_0 = copy.deepcopy(X_train_0)
target_train_copy_0 = copy.deepcopy(y_train_0)
my_input_0 = autograd.Variable(torch.from_numpy(data_train_copy_0).float())
my_target_0 = autograd.Variable(torch.from_numpy(target_train_copy_0).long())
#print("input : {}".format(my_input))

#transformer en pytorch tensor
data_train_copy_1 = copy.deepcopy(X_train_1)
target_train_copy_1 = copy.deepcopy(y_train_1)
my_input_1 = autograd.Variable(torch.from_numpy(data_train_copy_1).float())
my_target_1 = autograd.Variable(torch.from_numpy(target_train_copy_1).long())

#transformer en pytorch tensor
data_train_copy_2 = copy.deepcopy(X_train_2)
target_train_copy_2 = copy.deepcopy(y_train_2)
my_input_2 = autograd.Variable(torch.from_numpy(data_train_copy_2).float())
my_target_2 = autograd.Variable(torch.from_numpy(target_train_copy_2).long())


# test sur l'échantillon data_test
data_test_copy_0 = copy.deepcopy(X_test_0)
data_test_copy_1 = copy.deepcopy(X_test_1)
data_test_copy_2 = copy.deepcopy(X_test_2)
target_test_copy_0 = copy.deepcopy(y_test_0)
target_test_copy_1 = copy.deepcopy(y_test_1)
target_test_copy_2 = copy.deepcopy(y_test_2)
my_input_test_0 = autograd.Variable(torch.from_numpy(data_test_copy_0).float())
my_input_test_1 = autograd.Variable(torch.from_numpy(data_test_copy_1).float())
my_input_test_2 = autograd.Variable(torch.from_numpy(data_test_copy_2).float())
my_target_test_0 = autograd.Variable(torch.from_numpy(target_test_copy_0).long())
my_target_test_1 = autograd.Variable(torch.from_numpy(target_test_copy_1).long())
my_target_test_2 = autograd.Variable(torch.from_numpy(target_test_copy_2).long())


#definition du reseau simple 
class Net(nn.Module):
    def __init__(self, input_size, hidden_size,num_classes):
        super(Net, self).__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, num_classes)
        
        
    def forward(self, x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = F.softmax(x)
        return x

#instanciation d'un reseau       
model = Net(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
#optimizer params : parameters, lr : learning rate
opt = optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fct = nn.CrossEntropyLoss()
loss_tab_train_0 = []
f1_tab_train_0 = []
loss_tab_train_1 = []
f1_tab_train_1 = []
loss_tab_train_2 = []
f1_tab_train_2 = []

loss_tab_test_0 = []
f1_tab_test_0 = []
loss_tab_test_1 = []
f1_tab_test_1 = []
loss_tab_test_2 = []
f1_tab_test_2 = []

sc = lr_scheduler.CosineAnnealingLR(opt, 25, eta_min=0, last_epoch=-1)

# training loop
for epoch in range(number_of_epoch):
    print("period{}".format(epoch))
    out_0 = model(my_input_0)
    _, pred_0 = out_0.max(1)
#loss function
    loss_0 = loss_fct(out_0,my_target_0)
    out_1 = model(my_input_1)
    _, pred_1 = out_1.max(1)
#loss function
    loss_1 = loss_fct(out_1,my_target_1)
    out_2 = model(my_input_2)
    _, pred_2 = out_2.max(1)
#loss function
    loss_2 = loss_fct(out_2,my_target_2)
    #print('loss : {}'.format(loss.data[0]))
#zero the gradient
    model.zero_grad()
    loss_tab_train_0.append(loss_0.data.numpy())
    print(out_0.data.numpy()[0])
    print(my_target_0.data.numpy()[0])
    f1_tab_train_0.append(f1_score(my_target_0.data.numpy(),pred_0.data.numpy(),pos_label=1,average='binary'))
    loss_0.backward()
    loss_tab_train_1.append(loss_1.data.numpy())
    f1_tab_train_1.append(f1_score(my_target_1.data.numpy(),pred_1.data.numpy(),pos_label=1,average='binary'))
    loss_1.backward()
    loss_tab_train_2.append(loss_2.data.numpy())
    f1_tab_train_2.append(f1_score(my_target_2.data.numpy(),pred_2.data.numpy(),pos_label=1,average='binary'))
    loss_2.backward()
    #step of the gradient
    opt.step()

    out_0 = model(my_input_test_0)
    out_1 = model(my_input_test_1)
    out_2 = model(my_input_test_2)
    _, pred_0 = out_0.max(1)
    _, pred_1 = out_1.max(1)
    _, pred_2 = out_2.max(1)
    loss_test_0 = loss_fct(out_0,my_target_test_0)
    loss_test_1 = loss_fct(out_1,my_target_test_1)
    loss_test_2 = loss_fct(out_2,my_target_test_2)
    loss_tab_test_0.append(loss_test_0.data.numpy())
    loss_tab_test_1.append(loss_test_1.data.numpy())
    loss_tab_test_2.append(loss_test_2.data.numpy())
    print("loss_tab_test_0")
    print(loss_tab_test_0)
    print("loss_tab_test_1 :")
    print(loss_tab_test_1)
    print("loss_tab_test_2 :")
    print(loss_tab_test_2)
    f1_tab_test_0.append(f1_score(my_target_test_0.data.numpy(),pred_0.data.numpy(),pos_label=1,average='binary'))
    f1_tab_test_1.append(f1_score(my_target_test_1.data.numpy(),pred_1.data.numpy(),pos_label=1,average='binary'))
    f1_tab_test_2.append(f1_score(my_target_test_2.data.numpy(),pred_2.data.numpy(),pos_label=1,average='binary'))
"""    
print(len(loss_train_0))
print(len(loss_train_1))
print(len(loss_train_2))
print(len(loss_test_0))
print(len(loss_test_1))
print(len(loss_test_2))
"""

plotly.tools.set_credentials_file(username='celdeldel', api_key='vEP4v89BGCSdFEZ0p6Mq')


trace0 = go.Scatter(
    x=[i for i in range (number_of_epoch)],
    y=f1_tab_train_0
)

trace1 = go.Scatter(
    x=[i for i in range (number_of_epoch)],
    y=f1_tab_train_1
)

trace2 = go.Scatter(
    x=[i for i in range (number_of_epoch)],
    y=f1_tab_train_2
)

trace3 = go.Scatter(
    x=[i for i in range (number_of_epoch)],
    y=f1_tab_test_0
)

trace4 = go.Scatter(
    x=[i for i in range (number_of_epoch)],
    y=f1_tab_test_1
)

trace5 = go.Scatter(
    x=[i for i in range (number_of_epoch)],
    y=f1_tab_test_2
)

data = [trace0, trace3]

py.iplot(data, filename = 'basic net')