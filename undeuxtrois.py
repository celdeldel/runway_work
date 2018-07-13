#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:27:40 2018

@author: celdel

goal of this file : to have all the utilistary functions in one file 
"""

import face_recognition
import numpy as np
import copy


#renvoie la liste des lignes du fichier
def file_to_list(file_name):
    f = open(file_name,"r")
    l = f.read().split("\n")
    return l


#renvoie une liste copié avec les n premiuers elements a la fin   
def inverse_list(my_list,n):
       output=[]
       l = copy.deepcopy(my_list)
       l1 = l[0:n]
       l2 = l[n:]
       output = np.concatenate((l2, l1), axis=0)
       return output



 # renvoie l'encodings d'une image de nom img_name
def encodings_of_img(img_name):    
    return face_recognition.face_encodings(face_recognition.load_image_file(img_name))


#fonction qui renvoie un  tableau data_label_name_other de telle sorte que si le mannequin a été dans la marque a tester son score passe a 1 egaklelebt sur les photo des autresmarques
#attention : les images de la marque qui nous interessent doivent etre a la suite meme si pas forcement a la fin
#parametre est 0 si les images de la marque sont en premier et 1 sinon
def new_labels(data_labels_name,tolerance,parametre):
    n = sum(data_labels_name[1])
    help_list = []
    data_new = copy.deepcopy(data_labels_name[0])#on recopie les data
    names_new = copy.deepcopy(data_labels_name[2])
    a_data = []
    a_labels = []
    a_names = []
    #si on est dans le cas parametre = 0 on se ramene au cas ou parametre =1
    if(parametre == 0):
        #print("hemememe")
        
        a_data = inverse_list(data_labels_name[0],n)
        a_labels = inverse_list(data_labels_name[1],n)
        a_names = inverse_list(data_labels_name[2],n)
        #print("ok")
    else :
        a_data = copy.deepcopy(data_new)
        a_names = copy.deepcopy(names_new)
    data_tmtc=a_data[0:-n]
    data_brand=a_data[-n:]
    #print(a_names)
    l = np.zeros((len(data_tmtc),), dtype=int)
    
    for i in range(n):
        distances = face_recognition.face_distance(data_tmtc,data_brand[i])
        for j in range(len(data_tmtc)):
            if (distances[j] < tolerance):
                l[j]=1
                help_atom = [a_names[-n+i],a_names[j]]
                
                help_list.append(help_atom)
    l2 = np.ones((n,),dtype=int)
    if(parametre == 1):
        data_new_labels = np.concatenate((l, l2), axis=0)
    else :
        data_new_labels = np.concatenate((l2,l),axis=0)
    
    new_data_labels_names = [data_new,data_new_labels,names_new]

    return new_data_labels_names, help_list



#fonction qui renvoie un  tableau data_labels de telle sorte que si le mannequin a été dans la marque a tester son score passe a 1 egaklelebt sur les photo des autresmarques
#deux trois soucis remarques dans cette fonction
def new_labels_2(data_labels,tolerance):
    n = sum(data_labels[1])
    data_tmtc=data_labels[0][0:-n]
    data_brand=data_labels[0][-n:]
    l = np.zeros((len(data_tmtc),), dtype=int)
    for i in range(n):
        distances = face_recognition.face_distance(data_tmtc,data_brand[i])
        for j in range(len(data_tmtc)):
            if (distances[j]<tolerance):
                l[j]=1
    print(l)
    l2 = np.ones((n,),dtype=int)
    data_new_labels = np.concatenate((l, l2), axis=0)
    data_new = copy.deepcopy(data_labels[0])
    new_data_labels = [data_new,data_new_labels]
    return new_data_labels


#fonction qui prend un file en entree et 
#qui ecrit un tableau avec tous les encodings en chaque image
#qui renvoie un labeled &classified encoding tableau 
def www_l_c_encodings_from_file(my_file,str_brand):
    #initialisation de la list de output
    output = []
    output_e = []
    output_n = []
    output_c = []
    #recuperation des noms des images de my_file
    output_n_temp = file_to_list(my_file)
    #print(output_n_temp)
    #creation  des encodings pour chaque image et creation d'une premiere classification qui n'est pas definitive
    for n in output_n_temp:
        #recuperation des encodings
        try:
            e = encodings_of_img(n)
        except IOError:
            continue
        #cas ou il n'y a pas d'encodings
        if (len(e)==0):
            continue
        output_e.append(e[0])
        output_n.append(n)
        #classification temporelle
        if (n[10:25]==str_brand):
            c = 1
        else:
            c = 0
        output_c.append(c) 
    output_e = np.asarray(output_e)
    output_c = np.asarray(output_c)
    output_n = np.asarray(output_n)
        
    #new classification
    data_labels_names = [output_e,output_c,output_n]
    output = new_labels(data_labels_names,0.4,0)
    #output = [output_e, output_c[0], output_n, output_c[1]]
    #write in file
    
    return output


#fonction qui prend un file en entree et 
#qui ecrit un tableau avec tous les encodings en chaque image
#qui renvoie un labeled &classified encoding tableau 
#loads the data if needed my_file is : 'all_img_names.txt' and str-brand example : 'dries-van-noten'
"""
def www_l_c_encodings_from_file(my_file,str_brand):
    #initialisation de la list de output
    output = []
    output_e = []
    output_n = []
    output_c = []
    #recuperation des noms des images de my_file
    output_n_temp = file_to_list(my_file)
    #print(output_n_temp)
    #creation  des encodings pour chaque image et creation d'une premiere classification qui n'est pas definitive
    for n in output_n_temp:
        #recuperation des encodings
        try:
            e = encodings_of_img(n)
        except IOError:
            continue
        #cas ou il n'y a pas d'encodings
        if (len(e)==0):
            continue
        output_e.append(e[0])
        output_n.append(n)
        #classification temporelle
        if (n[5:20]==str_brand):
            c = 1
        else:
            c = 0
        output_c.append(c) 
    output_e = np.asarray(output_e)
    output_c = np.asarray(output_c)
    output_n = np.asarray(output_n)
        
    #new classification
    data_labels_names = [output_e,output_c,output_n]
    output = new_labels(data_labels_names,0.4,0)
    #output = [output_e, output_c[0], output_n, output_c[1]]
    #write in file
    
    return output

"""