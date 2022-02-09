# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 18:50:15 2021

@author: ceren
"""

from random import randrange
from csv import reader
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd


def csv_file_load(file):
    dataset = list()
    with open(file, 'r') as file:
        csv_r = reader(file)
        for row in csv_r:
            if not row:
                continue
            dataset.append(row)
    return dataset

def string2float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip()) 
        
def string2integer(dataset, column):
    valclass = [row[column] for row in dataset]
    list_of_class = []
    for i in range (6):
        list_of_class.append(valclass[i*720])
    dicitonary = dict()
    for i, value in enumerate(list_of_class):
        dicitonary[value] = i
    for row in dataset:
        row[column] = dicitonary[row[column]]
    return dicitonary

def gini_calculate(samples, list_of_class):
    instances_length = float(sum([len(g) for g in samples]))
    valgini = 0.0
    for g in samples:
        length = float(len(g))
        if length == 0:
            continue
        score_gini = 0.0
        for valclass in list_of_class:
            coefficient = [row[-1] for row in g].count(valclass) / length
            score_gini += coefficient * coefficient
        valgini += (1.0 - score_gini) * (length / instances_length)
    return valgini


def split_k_validation(dataset, k):
    partion_of_dataset = list()
    copy_set = list(dataset)
    foldsize = int(len(dataset) / k)
    for _ in range(k):
        fold_of_data = list()
        while len(fold_of_data) < foldsize:
            index = randrange(len(copy_set))
            fold_of_data.append(copy_set.pop(index))
        partion_of_dataset.append(fold_of_data)
    return partion_of_dataset

def test_split( i , threshold, dataset):
    dummy1 = list()
    dummy2 = list()
    for r in dataset:
        if r[i] < threshold:
            dummy1.append(r)
        else:
            dummy2.append(r)
    return dummy1, dummy2


def test_prun(groups_in_fold):
    dummy = [r[-1] for r in groups_in_fold]
    return max(set(dummy), key=dummy.count)

def fold_test(dataset , RandomForest, number_of_folds, *args):
    folds_of_dataset = split_k_validation(dataset, number_of_folds)
    accuracy_list = list()
    list_of_prediction = list()
    true_list = list()
    iterater=0
    for folds_in in folds_of_dataset:
       
        set_of_train = list(folds_of_dataset)
        set_of_train.remove(folds_in)
        set_of_train = sum(set_of_train, [])
        set_of_test = list()
        for r in folds_in:
            clone = list(r)
            set_of_test.append(clone)
            clone[-1] = None
        predicted_values = RandomForest(set_of_train, set_of_test, *args)
        real_values = [r[-1] for r in folds_in]
        accratio = 0
        for i in range(len(real_values)):
            if real_values[i] == predicted_values[i]:
                accratio = accratio + 1
        accuracy = float(accratio / float(len(real_values))) * 100
        accuracy_list.append(accuracy)
        true_list = true_list + real_values

        list_of_prediction = list_of_prediction + predicted_values
        iterater = iterater+1
   
    return accuracy_list, list_of_prediction, true_list


def node_split(dataset, feature):
    class_index = list(set(r[-1] for r in dataset))
    indexbounded = 62*4
    valbounded = 62*4
    scrbounded = 62*4
    boundGrp =  None
    feature_list = list()
    while len(feature_list) < feature:
        indices = randrange(len(dataset[0]) - 1)
        if indices not in feature_list:
            feature_list.append(indices)
    for i in feature_list:
        for r in dataset:
            listgrp = test_split(i, r[i], dataset)
            valgini = gini_calculate(listgrp, class_index)
            if valgini < scrbounded:
                indexbounded = i
                valbounded = r[i]
                scrbounded = valgini
                boundGrp = listgrp
    return {'index': indexbounded, 'value': valbounded, 'groups': boundGrp}


def split_nodes_all(nodes, maximum_depth , minimum_depth, features, tree_depth):
    L_Node, R_node = nodes['groups']
    del (nodes['groups'])
    if not L_Node or not R_node:
        nodes['dummy1'] = nodes['dummy2'] = test_prun(L_Node + R_node)
        return
    if tree_depth >= maximum_depth:
        nodes['dummy1'], nodes['dummy2'] = test_prun(L_Node), test_prun(R_node)
        return
    if len(L_Node) <= minimum_depth:
        nodes['dummy1'] = test_prun(L_Node)
    else:
        nodes['dummy1'] = node_split(L_Node, features)
        split_nodes_all(nodes['dummy1'], maximum_depth, minimum_depth, features, tree_depth + 1)
    if len(R_node) <= minimum_depth:
        nodes['dummy2'] = test_prun(R_node)
    else:
        nodes['dummy2'] = node_split(R_node, features)
        split_nodes_all(nodes['dummy2'], maximum_depth, minimum_depth, features, tree_depth + 1)


def tree_grow(set_of_train, maximum_depth, minimum_depth, features):
    root_of_tree = node_split(set_of_train, features)
    split_nodes_all(root_of_tree, maximum_depth, minimum_depth, features, 1)
    return root_of_tree


def bagging_subsample_forwardpass(forest_of_tree, r):
    predicted_values_list = [forwardpass(t, r) for t in forest_of_tree]
    return max(set(predicted_values_list), key=predicted_values_list.count)

def forwardpass(decision_node, index):
    if index[decision_node['index']] < decision_node['value']:
        if isinstance(decision_node['dummy1'], dict):
            return forwardpass(decision_node['dummy1'],index)
        else:
            return decision_node['dummy1']
    else:
        if isinstance(decision_node['dummy2'], dict):
            return forwardpass(decision_node['dummy2'], index)
        else:
            return decision_node['dummy2']

def bagging_subsample(dataset, partion_ratio):
    sampling = list()
    bag_size = round(len(dataset) * partion_ratio)
    while len(sampling) < bag_size:
        randomindex = randrange(len(dataset))
        sampling.append(dataset[randomindex])
    return sampling


def create_conf_matrix(predicated_classes, real_classes):
    conf_matrix = np.zeros([6, 6], dtype=int)
 
    for i in range(len(predicated_classes)):
        conf_matrix[int(real_classes[i]), int(predicated_classes[i])] += 1
    return conf_matrix


def compute_f_score(conf_matrix, class_index):
    true_positive = conf_matrix[class_index][class_index]
    ratio_of_recall = true_positive / np.sum(conf_matrix[class_index])
    ratio_of_precision = true_positive / np.sum(conf_matrix[:, class_index])
    print(ratio_of_recall)
    print(ratio_of_precision)
    score = 2 * ratio_of_precision * ratio_of_recall / (ratio_of_precision + ratio_of_recall+0.00000000001)
    return score

depth_accuracy =  np.zeros(30)
depth_f1 = np.zeros(30)

for tree in range (15,16):
    dataset = csv_file_load('./train_df.csv')
    dataset = csv_file_load('./train_df.csv')
    dataset = dataset[1:]
    
    for i in range(6):
    
        for j in range(720):
           dataset[720 * i:720 * (i + 1)][j][-1] = i

        
    for i in range(len(dataset)):
        dataset[i].pop(0)

    for i in range(0, len(dataset[0]) - 1):
        string2float(dataset, i)
    string2integer(dataset, len(dataset[0]) - 1)
   
    k_folds = 10
    maximum_depth = 15
    minimum_depth = 1
    feature = int(sqrt(len(dataset[0]) - 1))
    
    
    f1_scores, estimations, real_value = fold_test(dataset, rnd_frst, k_folds, maximum_depth, minimum_depth, sampling_ratio,
                                                      tree, feature)
    confusion_matrix = create_conf_matrix(estimations, real_value)
    score = 0
    accuracies = 0
    conf_matrix_all = 0
    for i in range(6):
        score += compute_f_score(confusion_matrix, i)
        accuracies += (720*5 + 2*confusion_matrix[i, i] - np.sum(confusion_matrix[:, i]))/4320
        conf_matrix_all += confusion_matrix[i, i]
    av_score = score / 6
    accuracies = accuracies/6
    conf_matrix_all = conf_matrix_all/720

    instruments = 'dog_bark jackhammer engine_idling children_playing street_music air_conditioner'.split()
    heatmap_conf_matrix = pd.DataFrame(confusion_matrix, index=[i for i in instruments], columns=[i for i in instruments])
    sn.set(font_scale=1)
    sn.heatmap(heatmap_conf_matrix, annot=True)
    plt.show()
    depth_accuracy[tree-15] =   (sum(f1_scores) / float(len(f1_scores)))
    depth_f1[tree-15] = av_score
