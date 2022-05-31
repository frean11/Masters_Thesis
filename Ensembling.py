# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 08:31:47 2022

@author: frede
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))

from Helper_Functions.Custom_functions import evaluation_functions

#%% Loading answers from dermotologists and subsetting with only relevant participants

truth_path = 'Evaluation data/Diagnoses.xls'
guess_path = 'Evaluation data/Full database anonymous.xlsx'

data, useful_participants, _, _ = evaluation_functions.csv_import(truth_path, guess_path)
data = data[data['OBSERVER ID'].isin(useful_participants)]

# Getting the correct labels and image name for each picture

im_path = 'Evaluation data/Skin cancer images'
im_shape = (256,256) # This is arbitrary as we only use the labels and image indexes

_, y_eval, idx = evaluation_functions.import_X_and_y(im_path, im_shape, data)
idx = [int(x) for x in idx]

# Loading predictions from CNN

prediction_path = 'Trained models/Multiclass-deep/Multiclass-deep/dropout_0.4/eval_files/eval_target.csv'

predictions = pd.read_csv(prediction_path)
pred = [x for x in predictions['prediction']]

# Defining number of iterations for each ensemble average

n_it = 1000

#%% Comparing human ensemble performances to human and algorithm baseline

# Human average performance

x_values, x_avg, y_values, y_avg = evaluation_functions.human_fpr_tpr(data, useful_participants, idx)
plt.scatter(x_avg, y_avg, c = 'blue', label = 'Human baseline')

# Algorithm baseline

df, auc = evaluation_functions.ROC_eval(pred, y_eval)
plt.plot(df['False positive rate'],df['True positive rate'], c = 'blue', label = 'CNN baseline')

# 3-man ensemble strategy

x_avg_three, y_avg_three, third_opinions = evaluation_functions.human_fpr_tpr_three_ensemble(data, useful_participants, idx, y_eval, n_it)
plt.scatter(x_avg_three, y_avg_three, c = 'green', label = '3-man ensemble average')
human_ensemble_performance = (x_avg,y_avg)

print(f'For the three human ensemble, the fpr is {x_avg_three} and tpr is {y_avg_three}')

# Full ensemble strategy

x_avg, y_avg = evaluation_functions.human_fpr_tpr_full_ensemble(data, useful_participants, idx, y_eval)
plt.scatter(x_avg, y_avg, c = 'red', label = 'Full ensemble score')

plt.title('Human ensemble performances')
plt.legend(loc = 'lower right')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('Visualizations/Human_ensembles.jpg')
plt.show()

print(f'The average number of third opinions in the 3-man ensemble is: {third_opinions}')

#%% Creating inner and outer list and comparing human and CNN performance on those sets

s_list = [2.5,3,4,6,8]

for s in s_list:

    # Getting auc scores
    print(f'The following results relate to s={s}')
    
    df_full, auc = evaluation_functions.ROC_eval(pred, y_eval)
    print(f'AUC score for the full prediction list: {auc}')
    
    pred_inner, pred_outer, y_inner, y_outer, indexes_inner, indexes_outer, idx_inner, idx_outer = evaluation_functions.inner_outer(pred, s, y_eval, idx)
    
    df_outer, auc = evaluation_functions.ROC_eval(pred_outer, y_outer)
    print(f'AUC for the outer list: {auc}')
    
    df_inner, auc = evaluation_functions.ROC_eval(pred_inner, y_inner)
    print(f'AUC for the inner list: {auc}')
    
    # Plotting everything together
    
    plt.plot(df_inner['False positive rate'],df_inner['True positive rate'], c = 'orange', label = 'CNN inner list')
    plt.plot(df_outer['False positive rate'],df_outer['True positive rate'], c = 'violet', label = 'CNN outer list')
    plt.plot(df_full['False positive rate'],df_full['True positive rate'], c = 'blue', label = 'CNN baseline')
    
    for i in [(idx_inner, 'orange'), (idx_outer, 'violet'), (idx, 'blue')]:  
        if i[0] == idx_inner:
            label = 'Human inner list'
        elif i[0] == idx_outer:
            label = 'Human outer list'
        else:
            label = 'Human baseline'
        x_values, x_avg, y_values, y_avg = evaluation_functions.human_fpr_tpr(data, useful_participants, i[0])
        plt.scatter(x_avg,y_avg, c = i[1], label = label)
    
    plt.legend(loc = 'lower right')
    plt.title(f'Performance on inner and outer lists with s = {s}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(f'Visualizations/Inner_outer_{s}.jpg')
    plt.show()

#%% Generating average ROC/AUC analysis for each of the values of s

for s in s_list:

    pred_inner, pred_outer, y_inner, y_outer, indexes_inner, indexes_outer, idx_inner, idx_outer = evaluation_functions.inner_outer(pred, s, y_eval, idx)
    
    averages_inner, auc_inner = evaluation_functions.averaging_ROC(n_it, data, useful_participants, idx_inner, pred, y_eval)
    averages_outer, auc_outer = evaluation_functions.averaging_ROC(n_it, data, useful_participants, idx_outer, pred, y_eval)
    
    print(f'The following results relate to s = {s}')
    print(f'The average auc score substituting the inner list is {auc_inner}')
    print(f'The average auc score substituting the outer list is {auc_outer}')
    
    # And plotting together with the baseline function 
    
    plt.plot(averages_inner['False positive rate'], averages_inner['True positive rate'], c = 'orange', label = 'Inner substitution')
    plt.plot(averages_outer['False positive rate'], averages_outer['True positive rate'], c = 'violet', label = 'Outer substitution')
    plt.plot(df_full['False positive rate'], df_full['True positive rate'], c = 'blue', label = 'CNN baseline')
    plt.scatter(x_avg, y_avg, c = 'blue', label = 'Human baseline') # The average values are coming from the previous cell
    plt.legend()
    plt.title(f'Augmented hybrid ensemble for s = {s}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(f'Visualizations/Hybrid_augmented_{s}.jpg')
    plt.show()


#%% Calculating the majority vote hybrid ensemble results

df_full, thresholds, scoring, third_opinions = evaluation_functions.hybrid_majority_vote(data, pred, y_eval, idx, n_it)

# Identifying the threshold and related x and y values for the point with the lowest number of third opinions

min_third_opinions = np.min(third_opinions)
print(f'The minimum number of third opinions necessary is {min_third_opinions}')
min_third_opinions_threshold = thresholds[third_opinions.index(min_third_opinions)]

# Getting the x and y coordinates for this point and deleting it from the general scoring

x, y = scoring[min_third_opinions_threshold]
scoring.pop(min_third_opinions_threshold)

# Generating the value to be plotted

scoring_values = list(scoring.values())
x_values = [x[0] for x in scoring_values]
y_values = [y[1] for y in scoring_values]

# Identifying the threshold and number of third opinions for the point with similar FPR and the point with similar TPR as the three human ensemble.

res = min(enumerate(x_values), key=lambda x: abs(x_avg_three - x[1]))
print(f'The number of third opinions to match the three human ensemple FPR score is {third_opinions[res[0]]}')

res = min(enumerate(y_values), key=lambda x: abs(y_avg_three - x[1]))
print(f'The number of third opinions to match the three human ensemble TPR score is {third_opinions[res[0]]}')

# Plotting the values

plt.scatter(x_values, y_values, c = 'grey', alpha = 0.3, label = 'Hybrid ensembles')
plt.scatter(x, y, c = 'red', label = 'Least third opinions hybrid ensemble')
plt.scatter(human_ensemble_performance[0], human_ensemble_performance[1], c = 'green', label = '3-man human ensemble')

# Plotting the baseline human and CNN performance for comparison

x_values, x_avg, y_values, y_avg = evaluation_functions.human_fpr_tpr(data, useful_participants, idx)
plt.scatter(x_avg, y_avg, c = 'blue', label = 'Human baseline')
plt.plot(df_full['False positive rate'], df_full['True positive rate'], c = 'blue', label = 'CNN baseline')
plt.title('Majority vote ensemble')
plt.legend(loc = 'lower right')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('Visualizations/Hybrid_majority.jpg')
plt.show()



