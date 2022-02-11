#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:57:52 2021

@author: djangraw
"""
# Import packages
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

# %% Declare functions

# Plot rock paper scissors training data
def plot_raw_RPS_data(emg_data,action_sequence=None,fs=500):
    """
    Plot rock paper scissors training data with rectangle indicators behind each non-rest action.
    Offset each one so they're all visible.

    Parameters
    ----------
    emg_data : numpy array of size NxD, where N is timepoints and D is channels.
        EMG data.
    action_sequence : list or array of strings, optional
        Labels for the action taking place in each second. If none, uses default specified in Lab 5.
    fs : float, optional
        Sampling rate of the Arduino. The default is 500.

    Returns
    -------
    None.

    """
    
    # Declare defaults
    if action_sequence is None:
        print('Using default action sequence.')
        action_sequence = ['Rest','Rock','Rest','Paper','Rest','Scissors']*10
    # Declare colors for background rectangles
    colors = {'Rock':'r',
          'Paper':'g',
          'Scissors':'b'}
    # Declare other constants
    emg_time = np.arange(len(emg_data))/fs
    channel_count = emg_data.shape[1]

    # Set up figure with a subplot for each channel
    fig,axes = plt.subplots(channel_count,1,figsize=(12,4),sharex=True,sharey=True)
    # Plot each channel in its own axis
    for channel_index in range(channel_count):
        # select axis
        plt.sca(axes[channel_index])
        plt.cla()
        # plot channel data
        plt.plot(emg_time,emg_data[:,channel_index])

        # Plot actions as rectangles
        for action_index,action in enumerate(action_sequence):
            if action != 'Rest':
                # plot a rectangle that is translucent and behind lines
                rect = patches.Rectangle([action_index,2], 1, 1.5, 
                                         facecolor=colors[action],
                                         alpha=0.5,zorder=-3) 
                plt.gca().add_patch(rect) # place rectangle on plot
                plt.text(action_index,2.1,action[0]) # add R,P, or S text
            
        # annotate plot
        plt.ylim([2,3])
        plt.xlabel('Time (s)')
        plt.ylabel(f'Ch{channel_index} (V)')
    # Annotate figure
    plt.tight_layout()



# Load a given set of EMG data and plot them if requested.
def load_rps_data(in_folder = '.', participant_list = ['A','B','C','D','E'], out_folder = '.', action_sequence=None):
    """
    Load the rock paper scissors data from multiple participants/files.
    Files must be saved as {in_folder}/{participant_list[i]_RPS_Data.npy .

    Parameters
    ----------
    in_folder : string, optional
        Path to folder where the data files sit. The default is '.'.
    participant_list : list or array of strings, optional
        The participants whose data you want to load. The default is ['A','B','C','D','E'].
    out_folder : string, optional
        Path to folder where the plots should go. The default is '.'. If None, no plots are produced.
    action_sequence : list or array of strings, optional
        Labels for the action taking place in each second. If None, uses default specified in Lab 5.

    Returns
    -------
    emg_data_list : list of M numpy arrays of size NxD, where M is the number of participants, N is timepoints and D is channels.
        EMG data for each of the participants.

    """
    print(f'Loading data from {len(participant_list)} participants...')
    emg_data_list = []
    for participant in participant_list:
        # load data
        filename = f'{in_folder}/{participant}_RPS_Data.npy'
        print(f'   Loading {filename}...')
        emg_data = np.load(filename)
        emg_data_list += [emg_data]
        # plot data and save figures if out_folder is not None
        if out_folder is not None:
            plot_raw_RPS_data(emg_data,action_sequence) # plot data
            plt.suptitle(f'RPS data for {participant}') # add figure title
            plt.savefig(f'{out_folder}/{participant}_RPS_Data.png') # save the figure tou out_folder
            plt.close() # close the figure that was created
    # return list of EMG data arrays
    return emg_data_list




# %% Sample usage

# If [A-E]_RPS_Data.npy are in the directory RPS_Data, the command below will 
# load their into the variable emg_data_list and save figures to the current
# directory.
emg_data_list = load_rps_data( 
                              participant_list = ['Doherty'], 
                              out_folder = '.')
