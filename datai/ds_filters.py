# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:05:40 2018

@author: Mathieu
"""
import numpy as np

def ds_2CAT (x, threshold=19):
    if 0<= x <= threshold :
        return 'Not depressive'
    elif x> threshold :
        return "Depressive"
    else :
        return np.nan

def ds_4CAT (x, threshold = [10, 19, 29]):
    if 0<=x <= threshold[0] :
        return 'not / minimal'
    
    elif threshold[0] <x and x <= threshold[1]:
        return "mild"
    
    elif threshold[1] < x<= threshold[2] :
        return "moderate"
    
    elif x> threshold[2]:
        return "severe"
    else :
        return np.nan

def ds_cs ( x, threshold = 19 ):
    #Gets minimal and mild patients off the batch
    #Lets undersample
    if x == 0:
        return 'Not depressive'
    elif x > threshold :
        return 'Depressive'
    else :
        return np.nan

    
#----------------------------------

#Filter functions for binary classification

def ds_2CAT_b (x, threshold=19):
    if 0<=x <= threshold :
        return 0
    elif x> threshold :
        return 1
    else :
        return np.nan

def ds_cs_b ( x, threshold = 19 ):
    #Gets minimal and mild patients off the batch
    #Lets undersample
    if x == 0:
        return 0
    elif x > threshold :
        return 1
    else :
        return np.nan
    
def d_only_b ( x, threshold = 19 ):
    
    if x>= 19   :  return 1
    else        :  return np.nan


#----------------------------------

def gender_to_b(x):
    if x == 'Female': return 1
    elif x == 'Male' : return 0
    else : raise ValueError
