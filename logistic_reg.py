# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 22:38:04 2021

@author: ceren
"""

import seaborn as sn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def sig_function(r):
    r = np.array(r, dtype=np.float64)
    return 1 / (1 + np.exp(-r))


def probability(dataset, coef):
    return sig_function(dataset.dot(coef))


def loss_func(beta, reg_coef, data_set, output):
    estimation = sig_function(data_set @ beta)
    estimation[estimation == 1] = 0.999  
    
    loss  = sum(-output* np.log(estimation) - (1 - output) * np.log(1 - estimation)) + (reg_coef/2)*sum(np.transpose(beta) @ beta) 
    return loss / len(output)


def descent_solver(data_set, coef, output):
    gradient = (np.transpose(data_set) @ (probability(data_set, coef) - output)) / len(output)
    return gradient


def coef_update(dataset, output, test_set, test_output, learning_rate, reg_coef):
    n, m = np.shape(dataset)
    coef = np.zeros((m, 1))
    loss_test_list = list()
    loss_list = list()
   
    loss_actual = loss_func(coef, reg_coef, dataset, output)
    loss_actual_test = 5
    iter = 0
    while (loss_actual_test >  0.005) & (iter < 150):
        
        coef = coef - learning_rate * descent_solver(dataset, coef, output)
        loss_actual = loss_func(coef, reg_coef, dataset, output)
        loss_actual_test = loss_func(coef, reg_coef, test_set , test_output )
     
        loss_test_list.append(loss_actual_test)
        loss_list.append(loss_actual)
        iter += 1
   
    return coef,loss_test_list,loss_list


def hyperplane_construct(dataset ,test_set , classNo ,learning_rate, reg_coef):
    n, m = np.shape(dataset)
    coef_store = np.zeros((m, classNo))
        

    for i in range(classNo):
        output = np.zeros((4320, 1))
        output[720 * i:720 * (i + 1)] = 1
        test_output =np.zeros((540,1))
        test_output[90 * i:90 * (i + 1)] = 1
        coef,cost_test_list,cost_list = coef_update(dataset, output, test_set , test_output, learning_rate, reg_coef)
  
              
        plt.plot(cost_list  )
        plt.plot(cost_test_list  )
        test = mpatches.Patch(color='blue', label='Train Loss')
        train = mpatches.Patch(color='red', label='Validation Loss')
        plt.legend(handles=[train , test])
        plt.title( "Class " + str(i+1))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()                     
        
        coef_store[:, i] = coef[:, 0]
    return coef_store


def classifier(dataset, coef_store, classNo):
    probabilities = probability(dataset, coef_store)
    predict = np.argmax(probabilities, axis=1)
    decision = np.mod(predict, classNo)
    return decision


def conf_construct(algoEst, realOut, classNumberX=6):
    confusionM = np.zeros([classNumberX, classNumberX], dtype=int)
    for i in range(len(algoEst)):
        confusionM[int(realOut[i]), int(algoEst[i])] += 1
    return confusionM

def FScoreCal(Mconfusion, NoClass):
    TruePos = Mconfusion[NoClass][NoClass]

    ValoRecall = TruePos / np.sum(Mconfusion[NoClass])
    ValoPrecision = TruePos / np.sum(Mconfusion[:, NoClass])
    F1 = 2 * ValoPrecision * ValoRecall / (ValoPrecision + ValoRecall+ 0.0000000000001)
    return F1

# =============================================================================

training_set = pd.read_csv("./train_df.csv")
attribute = list(training_set)
training_set = training_set.values
training_set = np.array(training_set[:, 1:26], dtype=np.float64)

for i in range(25):
    training_set[:,i] = ( training_set[:,i] - np.nanmean(training_set[:,i] )) / np.std(training_set[:,i] )


test_set = pd.read_csv("./test_df.csv")
attribute_test = list(test_set)
test_set = test_set.values
test_set = np.array(test_set[:, 1:26], dtype=np.float64)


for i in range(25):
    test_set[:,i] = ( test_set[:,i] - np.nanmean(test_set[:,i] )) / np.std(test_set[:,i] )



for coef in range(1,2):
    actual_output = np.zeros([4320, 1], dtype=int)
    for i in range(6):
        actual_output[720 * i:720 * (i + 1)] = i # 720*6 = 4320
    
    test_actual_output = np.zeros([540, 1], dtype=int)
    for i in range(6):
        test_actual_output[90* i:90 *(i + 1)] = i   
    
    learning_rate = 0.08
    reg_coef = 1.2
    
    coef_vector = hyperplane_construct(training_set,test_set, 6 ,learning_rate, reg_coef)
    prediction = classifier(training_set, coef_vector, 6)
    estimates_test = classifier(test_set ,coef_vector, 6)
    
    confusion_train = conf_construct(prediction, actual_output)
    confusion_test = conf_construct(estimates_test , test_actual_output)
    
    
    instruments = 'dog_bark jackhammer engine_idling children_playing street_music air_conditioner'.split()
    df_cm_test = pd.DataFrame(confusion_test, index=[i for i in instruments], columns=[i for i in instruments])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm_test, annot=True)
    plt.title("Confusion matrix test")
    plt.show()
    
 
    df_cm = pd.DataFrame(confusion_train, index=[i for i in instruments], columns=[i for i in instruments])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion matrix training")
    plt.show()
    
    score = 0
    acc = 0
    cci = 0
    for i in range(5):
        score += FScoreCal(confusion_train, i)
        acc += (5*720 + 2*confusion_train[i, i] - np.sum(confusion_train[:, i]))/4320
        cci += confusion_train[i, i]
    av_score = score / 6
    acc = acc/6
    cci = cci/4320
    print('average F1 training score : ', av_score)
    print('accuracy (training) : %', 100*acc)

    
    score = 0
    acc = 0
    cci = 0
    for i in range(5):
        score += FScoreCal(confusion_test, i)
        acc += (90*5 + 2*confusion_test[i, i] - np.sum(confusion_test[:, i]))/720
        cci += confusion_test[i, i]
    av_score = score / 6
    acc = acc/6
    cci = cci/720
    print('average F1 test score : ',av_score)
    print('accuracy (test) : %', 100*acc)


    