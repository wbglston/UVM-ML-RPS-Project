# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:51:51 2021
Project 3: EMG Rock, Paper, Scissors

@authors: James Doherty, Elliott Gear, Will Bigglestone
"""
# %% Import Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import math

# %% load Data, epoch data, extract features
#Import lab5 functions to do these
import Lab5_functions as lab5

emg_time, emg_data = lab5.load_data("A")
emg_epoch = lab5.epoch_data(emg_data)
epoch_features = lab5.extract_features(emg_epoch)

#define action sequence
sequence_actions = ['rest', 'rock','rest','paper','rest','scissors']

#call function for setting truth labels
instructed_action = lab5.make_truth_data(sequence_actions) # 1 epoch per action


used_actions = ['rest','rock','paper','scissors']

[features_ps,truth_labels_ps] = lab5.crop_ml_inputs(epoch_features, instructed_action,used_actions)


# %% Perform train-test split

#split data into train and test sets
X_train, X_test,y_train,y_test = train_test_split(features_ps, truth_labels_ps)

#declare hyperparameters
C = 1
gamma = 0.1

def evaluate_classifier(C,gamma,X_train,y_train,X_test,y_test):
    #create classifier object
    clf = SVC(C=C,gamma=gamma)
    
    # train or fit classifier to our training data
    clf.fit(X_train,y_train)
    
    y_prime_train = clf.predict(X_train)
    accuracy_train = np.mean(y_prime_train == y_train)
    
    #test the classifier on testing data
    y_prime_test = clf.predict(X_test)
    #get accuracy on test set
    accuracy_test = np.mean(y_prime_test == y_test)
    print(f"gamma={gamma},C={C},accuracy_train = {accuracy_train}, accuracy_test = {accuracy_test}")
    return(accuracy_train, accuracy_test)



C_array = np.array([1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6])
gamma = 0.1

#create arrays to receive accuracy values
accuracy_train = np.zeros(len(C_array))
accuracy_test = np.zeros(len(C_array))

#loop through C values
for C_index,C in enumerate(C_array):
    #call function to evaluate classifier training and testing accuracy
    accuracy_train[C_index], accuracy_test[C_index] = evaluate_classifier(C, gamma, X_train, y_train, X_test, y_test)

plt.figure()
plt.clf()
plt.plot(C_array, accuracy_train, label='training accuracy')
plt.plot(C_array, accuracy_test, label='testing accuracy')
plt.xlabel('regularization hyperparameter C')
plt.ylabel('accuracy (0-1)')
plt.xscale('log')
plt.legend()
plt.grid(True)

# %% Grid Search
from sklearn.model_selection import GridSearchCV

#declare our hyperparamteter grid
param_grid = {'C':C_array}
#create cross validated grid search object
clf = GridSearchCV(SVC(gamma=gamma), param_grid, scoring='accuracy')
print('Tuning hyperparameters...')
#perform cross validated grid search on our training set
clf.fit(X_train,y_train)
print('Done')

# %% Accuracy / Confusion Matrix 

plot_confusion_matrix(clf,X_test,y_test)

# %%test on another set
B_emg_time, B_emg_data = lab5.load_data("C")
B_emg_epoch = lab5.epoch_data(B_emg_data)
B_epoch_features = lab5.extract_features(B_emg_epoch)

#call function to crop for only paper and scissors data
[B_features_ps,B_truth_labels_ps] = lab5.crop_ml_inputs(B_epoch_features, instructed_action,used_actions)

clf_best = clf.best_estimator_

plot_confusion_matrix(clf_best,B_features_ps,B_truth_labels_ps)

# %% Save classifier
from joblib import dump,load

dump(clf_best,'Project3Classifier.joblib')

clf_best_loaded = load('Project3Classifier.joblib')
