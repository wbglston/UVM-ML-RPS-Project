# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:08:36 2021

@author: wbgls
"""

#%% Import Libraries

import numpy as np
from matplotlib import pyplot as plt
import math
#sklearn imports are done as so since they did not run as just importing sklearn
from sklearn import preprocessing
from sklearn import svm
#%% Part 2: Extract the Features and Labels

# Global definitions/defaults to be used throughout functions
FS = 500
EPOCH_DURATION = 1
EPOCHS_PER_ACTION = 1 


# Load data into file with this function

def load_data(file_name):
    """
    This function loads in data based on last name according to 
    rock-paper-scissors EMG data naming conventions set for the lab

    Parameters
    ----------
    file_name: last name associated with desired file

    Returns
    -------
    emg_data: emg data loaded in
    emg_time: time data loaded in
    
    """
    emg_time = np.load('./' + file_name + '_RPS_Time.npy')
    emg_data = np.load('./' + file_name + '_RPS_Data.npy')
    
    return emg_time, emg_data

def epoch_data(emg_data, fs = 500, epoch_duration = 1, channels = 3):

    #Uses reshape function to turn 2D EMG data into 3D, epoched EMG data
    
    # subtract mean from each channel
    for i in range(0,channels):
        emg_data[:,i] = emg_data[:,i] - np.mean(emg_data[:,i])
        
    #epoch_count is the final time divided by the time-length of each epoch
    final_time = len(emg_data)/fs
    epoch_count = math.ceil(final_time/epoch_duration)
    #epoch_sample_count is the total number of samples divided by number of epochs
    epoch_sample_count = math.ceil(len(emg_data)/epoch_count)
    #epoch emg_voltage
    emg_epoch = emg_data.reshape(epoch_count,epoch_sample_count,channels)
    #save the epoched data
    np.save('./emg_epoch',emg_epoch)
    return emg_epoch

def extract_features(emg_epoch, channels = 3):
    
    #FEATURE 1: VARIANCE
    epoch_quant = len(emg_epoch)
    #create 60x3 matrix of epoch variances for each feature
    emg_epoch_var = np.var(emg_epoch, axis=1) #Calculate the variance between epochs
    
    #FEATURE 2: MEAN ABSOLUTE VALUE
    #initialize MAV matrix, also 60x3
    emg_epoch_MAV = np.zeros([epoch_quant,channels])
    devsum = 0
    
    
    for i in range(len(emg_epoch[1,1,:])):
        for j in range(len(emg_epoch[1,:,1])):
            #mean of one entire epoch
            avg = np.mean(emg_epoch[1:,j,i])
            #Absolute deviations of epoch
            dev = np.absolute(emg_epoch[:,j,i] - avg)
            devsum = devsum + dev
            emg_epoch_MAV[:,i] = devsum/len(emg_epoch[1,:,1])
            
            
    #FEATURE 3: ZERO CROSSING
    #Last 60x3 matrix
    zero_crossing = np.zeros([epoch_quant,channels])
    zc = 0
    
    
    for i in range(len(emg_epoch[1,1,:])):
        for j in range(len(emg_epoch[:,1,1])):
        
            zc = np.where(np.diff(np.signbit(emg_epoch[j,:,i])))[0]
            zero_crossing[j,i] = len(zc)
    
    epoch_features = np.concatenate((emg_epoch_var,emg_epoch_MAV,zero_crossing),\
                                  axis=1)
    
    epoch_features = preprocessing.scale(epoch_features)
    
    return epoch_features

def make_truth_data(sequence_actions, epochs_per_action = 1):
    """
    This function uses the action sequence to place proper labels in array 
    that align with each epoch for labeling, ensuring to use the same label
    multiple times if multiple epochs contain the same action

    Parameters
    ----------
    sequence_actions: the sequence the actions were performed in 
    epochs_per_action: number of epochs in a row with the same action

    Returns
    -------
    instructed_action: array of actions for each epoch
    
    """
    #initialize instructed actions
    instructed_action = np.empty(len(sequence_actions*epochs_per_action*10),dtype = object)
    #loop through as many instructed actions
    for action_index in range(len(instructed_action)):
        #loop through how many epochs occur with same action
        for sample_epoch in range(epochs_per_action):
            #match instructed action with appropriate sequenced action
            instructed_action[action_index + sample_epoch] = sequence_actions[action_index % 6]
        
    return instructed_action

def crop_ml_inputs(feature_matrix,instructed_action,truth_labels):
    """
    This function crops the feature matrix to only include the actions being
    considered

    Parameters
    ----------
    feautre_matrix: full feature matrix with all labels
    instructed_action: full array of labels of all epochs
    truth_labels: desired labels for data being considered

    Returns
    -------
    features_ps: cropped feature matrix only containing data with proper label
    truth_labels_ps: cropped labels that only match labels being used
    
    """
    #initialize cropped version of feature matrix
    features_crop = np.zeros(len(feature_matrix),bool)
    #loop through feature matrix
    for crop_index in range(len(feature_matrix)):
        #check if current instructed action matches desired truth label
        features_crop[crop_index] = np.isin(instructed_action[crop_index],truth_labels)
    
    #delete feature and truth label data that does not match desired labels
    features_ps = np.delete(feature_matrix, np.argwhere(features_crop != True),axis = 0)
    truth_labels_ps = np.delete(instructed_action, np.argwhere(features_crop != True), axis = 0)
    
    return features_ps, truth_labels_ps 

def scatter_plot(column_subset,shorthand_names,truth_labels):
    """
    This function creates a scatter plot using a subset of the features and 
    the names of said features comparing the two, with a truth matrix to
    ensure proper labeling

    Parameters
    ----------
    column_subset: the subset of the feature matrix being plotted
    shorthand_names: the names of the features being plotted
    truth_labels: the truth matrix that labels the examples to the correct action
    
    """
    # define intial features with indexing 
    feature0 = column_subset[:,0] 
    feature1 = column_subset[:,1]
    # make a scatter plot
    plt.figure()
    # set up truth labels in plot
    plt.plot(feature0[truth_labels =='paper'],feature1[truth_labels =='paper'],'o',label = 'paper')
    plt.plot(feature0[truth_labels =='scissors'],feature1[truth_labels =='scissors'], 'o',label = 'scissors')
    # add graph information
    plt.xlabel(shorthand_names[0])
    plt.ylabel(shorthand_names[1])
    # paper vs scissor title
    plt.title('Scatter plot in feature space')
    # include legend
    plt.legend()
    plt.savefig(f'lab5_{shorthand_names[0]}_vs_{shorthand_names[1]}')

def fit_classifier(features, labels):
    """
    This function uses features and labels to first set labels to a binary
    before then training a linear classifier with said data

    Parameters
    ----------
    features: features used for training classifier
    labels: labels used for training classifier

    Returns
    -------
    classifier: classifier trained by given data 
    
    """
    #initialize binary version of labels
    labels_binary = np.zeros(len(labels))
    #loop through all labels
    for label_index in range(len(labels)):
        #check if label is scissors
        if(labels[label_index] == 'scissors'):
            #scissors label replaced with 1
            labels_binary[label_index] = 1;
        else:
            #paper label replaced with -1
            labels_binary[label_index] = -1;
    #define linear classifier
    classifier = svm.LinearSVC()
    #fit classifier to features and labels
    classifier.fit(features,labels)
    return classifier

def predictor_histogram(trained_classifier, features, truth_labels):
    """
    This function uses the trained classifier to plot a histogram comparing
    the two labels with a threshold of 0

    Parameters
    ----------
    trained_classifier: classifier used on example data to make histogram
    features: data that classifier forms histogram with
    truth_labels: labels for differentiating different histogram data

    """
    #use predictor on dataset
    z_prime = features @ trained_classifier.coef_ . T + trained_classifier.intercept_
    plt.figure()
    #plot histogram for first label
    plt.hist(z_prime[truth_labels == truth_labels[0]], alpha = 0.5, bins = 10, label=truth_labels[0])
    #plot histogram for second label
    plt.hist(z_prime[truth_labels == truth_labels[1]], alpha = 0.5, bins = 10, label=truth_labels[1])
    #plot threshold (bias term ensures this is 0)
    plt.vlines(x=0, ymin=0, ymax=4, label="threshold")
    #set up histogram
    plt.legend()
    plt.xlabel("Predictor Values")
    plt.ylabel("Number of Trials")
    plt.title("Predictor Histogram for Paper vs Scissors")

def evaluate_classifier(trained_classifier, feature_matrix, truth_labels):
    """
    This function uses the trained classifier to predict the labeling for 
    examples from a feature matrix. These predictions are compared to the
    actual labeling to make a confusion matrix and calculate accuracy.

    Parameters
    ----------
    trained_classifier: classifier used to predict labels for examples
    feature_matrix: matrix containing examples that can be used for predictions
    truth_labels: correct labeling of examples used for comparisons in confusion
    matrix and accuracy calculation
    
    """
    #predict labels for each feature using classifier
    trained_classifier.predict(feature_matrix)
    #use predictor on dataset
    z_prime = feature_matrix @ trained_classifier.coef_ . T + trained_classifier.intercept_
    #initialize confusion matrix (assumes 2 labels)
    confusion_matrix = np.zeros((2,2))
    
    #each matrix value is sum of cases where sign of example is found (since 
    #threhold is 0) for appropriate label to compare prediction and truth
    #predicted paper, actually paper
    confusion_matrix[0,0] = np.sum((np.sign(z_prime[truth_labels == truth_labels[0]]) == -1))
    #predicted paper, actually scissors
    confusion_matrix[0,1] = np.sum((np.sign(z_prime[truth_labels == truth_labels[1]]) == -1))
    #predicted scissors, actually paper
    confusion_matrix[1,0] = np.sum((np.sign(z_prime[truth_labels == truth_labels[0]]) == 1))
    #predicted scissors, actually scissors
    confusion_matrix[1,1] = np.sum((np.sign(z_prime[truth_labels == truth_labels[1]]) == 1))
    
    #plot confusion matrix
    plt.figure()
    plt.pcolor(confusion_matrix)
    #set up plot properly with colorbar
    plt.colorbar(label = '# of Trials')
    plt.title("ML Prediction")
    plt.ylabel('Prediciton Action')
    plt.xlabel('Actual Action')
    plt.yticks([0.5,1.5],["paper","scissors"])
    plt.xticks([0.5,1.5],["paper","scissors"],ha="center")
    
    #calculate accuracy of predictions (out of 100)
    accuracy = 100*(confusion_matrix[0,0] + confusion_matrix[1,1]) / (np.sum(confusion_matrix))
    #print resulting accuracy
    print("Classifier accuracy is " + str(accuracy))

