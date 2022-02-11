#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time
import pyautogui
import serial
import numpy as np
import Lab5_functions as lab5
from joblib import load

#%% Part C
def click_icon(icon):
    screen_width, screen_height = pyautogui.size()    
    # due to the top ribbon of browswers i found screen_height/1.8 to be 
    # most optimal. These fractions have been tested on Elliott's Laptop.
    # desktop and Will's laptop. Web browswer window must be filling
    # entire screen
    if icon == 1:
        print('clicking rock')
        pyautogui.moveTo(screen_width/5.9, screen_height/1.8)
        pyautogui.click()
        
    if icon == 2:
        print('clicking paper')
        pyautogui.moveTo(screen_width/3.4, screen_height/1.8)
        pyautogui.click()
        
    if icon == 3:
        print('clicking scisors')
        pyautogui.moveTo(screen_width/2.3, screen_height/1.8)
        pyautogui.click()
        
def read_emg(arduinoData):
    # read a line from serial port
    arduinoString = arduinoData.readline()
    
    # clean the output
    arduinoList = arduinoString.split()
        
    # convert to volt 
    # not sure why we had to handle this error but here we are
    try:
        arduinoList[1] = int(arduinoList[1])*5.0/1023
        arduinoList[2] = int(arduinoList[2])*5.0/1023
        arduinoList[3] = int(arduinoList[3])*5.0/1023
        arduinoList[4] = int(arduinoList[4])*5.0/1023
    except IndexError as error:
        print('oops trying again')
    
    # we are done with this list here, i think return the 4 values
    try:
        return arduinoList[0], arduinoList[1], arduinoList[2], arduinoList[3], arduinoList[4]
    except IndexError as error2:
        print('data read failed')
    

    
def main(port = '/dev/cu.usbserial-1450'):
    FS = 500
    RECORDING_DURATION = 3
    N_CHANNELS = 3
    
    #divide the time array by the sampling frequency up to the recording duration
    sample_time = np.arange(0,RECORDING_DURATION,1/FS);
    #preallocate the sample data array with nan values
    sample_data = np.zeros((FS*RECORDING_DURATION,N_CHANNELS))*np.nan;
    
    # create com port
    # open serial port
    com_port = port
    arduino_data = serial.Serial(port = com_port,baudrate = 500000)
    arduino_data.close()
    #open the serial data for arduino
    arduino_data = serial.Serial(port=com_port,baudrate=500000)
    
    # read emg 
    
    [sample_time, sample_data[sample_index,0], sample_data[sample_index,1], sample_data[sample_index,2]] = read_emg(arduino_data)
    
    # load / create classifier object
    
    classifier = load("Project3Classifier.joblib")
    
    emg_epoch = lab5.epoch_data(sample_data)
    epoch_features = lab5.extract_features(emg_epoch)

    y_test = classifier.predict(epoch_features)
    
    return y_test
    
y_test = main("COM8")
click_icon(1)