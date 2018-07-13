#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:25:08 2018

@author: celdel

knn qui affiche les scores sur plotly
"""

import undeuxtrois
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import os
import undeuxtrois


names_file = "all_img_names.txt"
str_brand =  'dries-van-noten'
name_file = "new_names.txt"

list_names = undeuxtrois.file_to_list(names_file)

new_list_names = []
fichier = open(name_file, "w")

for name in list_names:
    new_list_names.append("data/{}".format(name))
    fichier.write("data/{}".format(name))
    fichier.write("\n")

fichier.close()
enc = undeuxtrois.www_l_c_encodings_from_file(name_file,str_brand)[0]
print(enc)
enc_temp = enc

input_brut = enc_temp[0]
#print(input_brut)
input_labels = enc_temp[1]
#print(input_labels)

X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(input_brut,input_labels, test_size=0.20, random_state=123)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(input_brut,input_labels, test_size=0.26, random_state=123)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(input_brut,input_labels, test_size=0.33, random_state=123)

#test des vrai vrais 
def acc_score(target_test,predic_test):
    c = 0
    s = 0
    for i in range(len(predic_test)):
        if (target_test[i] == 1):
            s = s + 1.0
            if(predic_test[i]>0.5):
                c = c +1.0
    if s == 0: 
        return 0
    return c/s

# revoie une liste de predictions a partir de deux narray le premier est le training le deuxieme test
def cross_validation_process(data_train,target_train,data_test,target_test,fin):
    f1_score_test = []
    acc_true_pos =  []
    for i in range(fin):
        celine = KNeighborsClassifier(n_neighbors = i+1, weights='distance',metric='euclidean')
        celine.fit(data_train,target_train)
        l_test= celine.predict(data_test)
        f1_score_test.append(f1_score(target_test,l_test,pos_label=1,average='binary'))
        acc_true_pos.append(acc_score( target_test,l_test))
        
        
    return f1_score_test, acc_true_pos

print(X_test_0)
print(y_train_0)
print(X_test_0)
f1_0,acc_0 = cross_validation_process(X_train_0,y_train_0,X_test_0,y_test_0,50)
f1_1,acc_1 = cross_validation_process(X_train_1,y_train_1,X_test_1,y_test_1,50)
f1_2,acc_2 = cross_validation_process(X_train_2,y_train_2,X_test_2,y_test_2,50)

#aller sur plotly pour afficher, se creer un compte et recuperer une clé API
plotly.tools.set_credentials_file(username='celdeldel', api_key='XrahLix08lPZtAWw30Iz')

trace0 = go.Scatter(
    x=[i for i in range (50)],
    y=f1_0
)

trace1 = go.Scatter(
    x=[i for i in range (50)],
    y=acc_0
)

trace2 = go.Scatter(
    x=[i for i in range (50)],
    y=f1_1
)

trace3 = go.Scatter(
    x=[i for i in range (50)],
    y=acc_1
)

trace4 = go.Scatter(
    x=[i for i in range (100)],
    y=f1_2
)

trace5 = go.Scatter(
    x=[i for i in range (50)],
    y=acc_2
)

data = [trace0, trace1, trace2, trace3, trace4, trace5]

py.iplot(data, filename = 'knn classification')
