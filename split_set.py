# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:17:06 2021

@author: ceren
"""

import pandas as pd
import csv
import numpy as np
import random

np.random.seed(0)
random.seed(0)
np.random.seed(0)


df_all = pd.read_csv('./extractedSet.csv')
df_pca = pd.read_csv('./extractedPca.csv')



tasks = df_all['class'].unique()
tasks = list( tasks )
tasks.remove("gun_shot")
tasks.remove("car_horn")
# tasks.remove("siren")
# tasks.remove("drilling")


flag = True

i = 0

for task in tasks:

    task_dummy_org = df_all.loc[ df_all['class']  == task] 
    print("For " + task)
    print(len(task_dummy_org))
    

    task_dummy_pca = df_pca.loc[ df_pca['class']  == task] 
    print("For " + task)
    print(len(task_dummy_pca))   

    if( task != 'siren' ):
        task_dummy_org = task_dummy_org.iloc[:-100]
        task_dummy_pca = task_dummy_pca.iloc[-99:]

    else:
        task_dummy_org = task_dummy_org.iloc[:-29]
        task_dummy_pca = task_dummy_pca.iloc[-28:]
        
    print("~~~~~~~~")


    print("For " + task)
    print(len(task_dummy_org))
    
    print("For " + task)
    print(len(task_dummy_pca))   

    print("============")
    
    

    df_train = task_dummy_org.sample( frac = 0.80 )  
    df_pca_train = df_pca.loc[ df_train.index]
    
    df_notTrain = task_dummy_org.loc[ ~task_dummy_org.index.isin( df_train.index) ]
    df_pca_notTrain = df_pca.loc[ ~df_pca.index.isin( task_dummy_pca.index) ] 
    df_pca_notTrain = df_pca_notTrain.loc[ ~df_pca_notTrain.index.isin( df_train.index) ]
    
    df_test = df_notTrain.sample( frac = 0.50 )
    df_pca_test = df_pca_notTrain.loc[ df_test.index]
    

    df_valid = df_notTrain.loc[ ~df_notTrain.index.isin(df_test.index) ]
    df_pca_valid = df_pca_notTrain.loc[ df_valid.index ]


    if( flag):
        train_df = df_train.copy()
        validation_df = df_valid.copy()
        test_df = df_test.copy()
        train_df_pca = df_pca_train.copy()
        validation_df_pca = df_pca_valid.copy()
        test_df_pca = df_pca_test.copy()
        flag =False
    else:
        train_df = train_df.append( df_train , ignore_index = True )
        validation_df = validation_df.append( df_valid , ignore_index = True )
        test_df = test_df.append( df_test , ignore_index = True )
        
        train_df_pca = train_df_pca.append( df_pca_train , ignore_index = True )
        validation_df_pca = validation_df_pca.append( df_pca_valid , ignore_index = True )
        test_df_pca = test_df_pca.append( df_pca_test , ignore_index = True )

    # print("~~~~~~~~")


    # print("--For " + task)
    # print(len(train_df_pca))
    # print(len(validation_df_pca))
    # print(len(test_df_pca))

    # print("~~~~~~~~")
    
    # task_dummy_org = train_df.loc[ train_df['class']  == task] 
    # print("For " + task + " train")
    # print(len(task_dummy_org))
    # task_dummy_org = validation_df.loc[ validation_df['class']  == task] 
    # print("For " + task + " valid")
    # print(len(task_dummy_org))
    # task_dummy_org = test_df.loc[ test_df['class']  == task] 
    # print("For " + task + " test")
    # print(len(task_dummy_org))
    # print("--------------------PCA------------------------")
    
    # task_dummy_org = train_df_pca.loc[ train_df_pca['class']  == task] 
    # print("For " + task + " train")
    # print(len(task_dummy_org))
    # task_dummy_org = validation_df_pca.loc[ validation_df_pca['class']  == task] 
    # print("For " + task + " valid")
    # print(len(task_dummy_org))
    # task_dummy_org = test_df_pca.loc[ test_df_pca['class']  == task] 
    # print("For " + task + " test")
    # print(len(task_dummy_org))
    # print("--------------------------------------------")


for task_a in tasks:
    


    task_dummy_org = train_df.loc[ train_df['class']  == task] 
    
    print("For " + task + " train")
    print(len(task_dummy_org))
    task_dummy_org = validation_df.loc[ validation_df['class']  == task] 
    print("For " + task + " valid")
    print(len(task_dummy_org))
    task_dummy_org = test_df.loc[ test_df['class']  == task] 
    print("For " + task + " test")
    print(len(task_dummy_org))
    print("--------------------PCA------------------------")
    
    task_dummy_org = train_df_pca.loc[ train_df_pca['class']  == task] 
    print("For " + task + " train")
    print(len(task_dummy_org))
    task_dummy_org = validation_df_pca.loc[ validation_df_pca['class']  == task] 
    print("For " + task + " valid")
    print(len(task_dummy_org))
    task_dummy_org = test_df_pca.loc[ test_df_pca['class']  == task] 
    print("For " + task + " test")
    print(len(task_dummy_org))
    print("--------------------------------------------")
    
    i += 1
    

# base_dict = {k: v for v, k in enumerate(tasks)}
    
# train_df['class'] = train_df['class'].map(base_dict)
# validation_df['class'] = validation_df['class'].map(base_dict)
# test_df['class'] = test_df['class'].map(base_dict)

# train_df_pca['class'] = train_df_pca['class'].map(base_dict)
# validation_df_pca['class'] = validation_df_pca['class'].map(base_dict)
# test_df_pca['class'] = test_df_pca['class'].map(base_dict)

train_df.drop('class', axis=1)
validation_df.drop('class', axis=1)
test_df.drop('class', axis=1)

train_df_pca.drop('class', axis=1)
validation_df_pca.drop('class', axis=1)
test_df_pca.drop('class', axis=1)

train_df.to_csv("./train_df.csv")
validation_df.to_csv("./validation_df.csv")
test_df.to_csv("./test_df.csv")

train_df_pca.to_csv("./train_df_pca.csv")
validation_df_pca.to_csv("./validation_df_pca.csv")
test_df_pca.to_csv("./test_df_pca.csv")
