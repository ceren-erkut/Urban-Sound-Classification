#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 20:43:45 2021

@author: ceren
"""

# -*- coding: utf-8 -*-


#Importing relevant methods and classes
import numpy as np
from csv import reader
from math import sqrt
from random import randrange
import time
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# =============================================================================

# KNN Main Function
def knn_main(training_dataset, test_dataset, K_value):
    output_list = list()
    for row in test_dataset:
        nList = get_neighbors(training_dataset, row, K_value)
        out_val = [roww[-1] for roww in nList]
        output = max((out_val), key=out_val.count) 
        # prediction is obtained
        output_list.append(output)
    return (output_list)

# gets nearest neighbors
def get_neighbors(training_dataset, test_dataset, K_value):
    euclid_distance_list = list()
    for row in training_dataset:
        euclid_distance_value = calculate_euclid_distance(test_dataset, row)
        euclid_distance_list.append((row, euclid_distance_value))
    euclid_distance_list.sort(key=lambda tup: tup[1]) 
    neighbors_list = list()
    for i in range(1, K_value):
        neighbors_list.append(euclid_distance_list[i][0])
    return neighbors_list

# calculates the euclidian distance
def calculate_euclid_distance(test_dataset, row):
    euclid_distance = 0.0
    for i in range(len(test_dataset)-1):
        euclid_distance = euclid_distance + abs((test_dataset[i] - row[i])**2)
    euclid_distance = sqrt(euclid_distance)
    return euclid_distance

# =============================================================================

# loads csv files
def load_csv_file(file_name):
    dataset = list()
    with open(file_name, 'r') as file: 
        csvFile = reader(file)
        for row in csvFile:
            if not row:
                continue
            dataset.append(row)
    return dataset

# converts strings in the dataset to floats
def str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip()) 

# converts floats in the dataset to integers
def float_to_int(dataset, column, n):
    class_Numbers = [row[column] for row in dataset]
    class_Sequence = []
    for i in range (6):
        class_Sequence.append(class_Numbers[i*n])
    LUtable = dict()
    for i, value in enumerate(class_Sequence):
        LUtable[value] = i
    for row in dataset:
        row[column] = LUtable[row[column]]
    return LUtable

# normalizes dataset, between 0 and 1
def normalize_dataset(dataset):
    extreme_values = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        minimum_val = min(col_values)
        maximum_val = max(col_values)
        extreme_values.append([minimum_val, maximum_val])
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - extreme_values[i][0]) / (extreme_values[i][1] - extreme_values[i][0])

# =============================================================================


# Find the percentage of correctly classified values
def accuracy_metric(realOutabels, systemOutputs):
    correctlyClassified = 0
    for i in range(len(realOutabels)):
        if realOutabels[i] == systemOutputs[i]:
            correctlyClassified = correctlyClassified + 1

    return float(correctlyClassified / float(len(realOutabels))) * 100


# F-score Calculation
def calculate_f_score(confusion_matrix, class_num):
    true_positives = confusion_matrix[class_num][class_num]
    true_positive_ratio = true_positives / np.sum(confusion_matrix[class_num])
    true_positive_precision = true_positives / np.sum(confusion_matrix[:, class_num])
    f_score = 2 * true_positive_precision * true_positive_ratio / (true_positive_precision + true_positive_ratio)
    return f_score


def tpr_fpr(confusion_matrix, class_num=6):
    true_positives, FalsePos, TrueNeg, FalseNeg = 0, 0, 0, 0
    true_positives = confusion_matrix[class_num][class_num]
    for i in (0,1,2,3,4):
        FalsePos = FalsePos + confusion_matrix[i][class_num]
        FalseNeg = FalseNeg + confusion_matrix[class_num][i]
    TrueNeg = 4320 - FalsePos - FalseNeg - true_positives
    return true_positives, FalsePos, TrueNeg, FalseNeg


def pre_rec(confusion_matrix, class_num=6):
    true_positives, FalsePos, TrueNeg, FalseNeg = 0, 0, 0, 0
    true_positives = confusion_matrix[class_num][class_num]
    for i in (0, 6):
        FalsePos = FalsePos + confusion_matrix[i][class_num]
        FalseNeg = FalseNeg + confusion_matrix[class_num][i]
    TrueNeg = 4320 - FalsePos - FalseNeg - true_positives
    ValoRecall = true_positives / (true_positives+FalseNeg)
    ValoPrecision = true_positives / (true_positives+FalsePos)
    return ValoRecall, ValoPrecision


# =============================================================================

start = time.time()

class_num = 6
test_instance_per_class = 90
training_instance_per_class = 120

main_dataset = load_csv_file('./train_df_pca.csv')
main_dataset = main_dataset[1:]

test_dataset = load_csv_file('./test_df_pca.csv')
test_dataset = test_dataset[1:]

length_dataset = len(main_dataset[0]) 
length_test_dataset = len(test_dataset[0]) 


# =============================================================================

   
for i in range(length_dataset - 1):
    str_to_float(main_dataset, i)

float_to_int(main_dataset, length_dataset-1, class_num*training_instance_per_class)

true_test_output = np.zeros([class_num*test_instance_per_class, 1], dtype=int)
for i in range(class_num):
    true_test_output[test_instance_per_class * i : test_instance_per_class * (i + 1)] = i

for i in range(length_test_dataset - 1):
    str_to_float(test_dataset, i)

float_to_int(test_dataset, length_test_dataset-1, test_instance_per_class)

for i in range(len(test_dataset)):
    test_dataset[i].pop(0)
    
for i in range(len(main_dataset)):
    main_dataset[i].pop(0)


accuracy_values = np.zeros(16)
accuracy_values_diag = np.zeros(16)
f_score_values = np.zeros(16)

for selected_k_value in range (2,18):
    
    print(selected_k_value)
    normalize_dataset(main_dataset)
    knn_output = knn_main(main_dataset, test_dataset, selected_k_value)
    
    predicted_test_output = (np.array(knn_output))*5
    
    # calculates confusion matrix
    confusion_matrix = np.zeros([class_num, class_num], dtype=int)
    for i in range(len(predicted_test_output)):
        confusion_matrix[int(true_test_output[i]), int(predicted_test_output[i])] += 1     
    print("Confusion Matrix for K = ", selected_k_value)
    print(confusion_matrix)
    
    # f score 
    f_score_total = 0
    accuracy_first = 0
    accuracy_second = 0
    for i in range(class_num):
        f_score_total += calculate_f_score(confusion_matrix, i)
        accuracy_first += (test_instance_per_class*5 + 2*confusion_matrix[i, i] - np.sum(confusion_matrix[:, i])) / (class_num*test_instance_per_class)
        accuracy_second += confusion_matrix[i, i] / (class_num*test_instance_per_class)
    FScore6 = f_score_total / class_num
    print('Average F score is ', calculate_f_score(confusion_matrix, 0))
    accuracy_first = 100*accuracy_first / class_num
    print('Accuracy : %', accuracy_first)
    accuracy_second = 100*accuracy_second
    print('Correct Classification: %', accuracy_second)
    instruments = 'dog_bark jackhammer engine_idling children_playing street_music air_conditioner'.split()
    confmatvals = pd.DataFrame(confusion_matrix, index=[i for i in instruments], columns=[i for i in instruments])
    sn.set(font_scale=1)
    sn.heatmap(confmatvals,  annot=True)
    plt.show()
    accuracy_values[selected_k_value-2] = accuracy_first
    accuracy_values_diag[selected_k_value-2] = accuracy_second
    f_score_values[selected_k_value-2] = FScore6
    print("K-value Accuracies")
    print(accuracy_values)
    print()
    print(accuracy_values_diag)
    print()
    print(f_score_values)
    