#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:56:18 2019

@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""

import pickle
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import os,sys
import pandas as pd
import shutil

sys.path.append('./datai')
sys.path.append('./pablo')
sys.path.append('./ml_run_lib')

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import pkgutil
search_path = ['.'] # set to None to see all modules importable from sys.path
all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
#print(all_modules)

#from datai import datasets as ds
from datai import scorer

#from ml_run_lib import check_all_c as cac
from ml_run_lib import experiment_functions
#from ml_run_lib import massive_grid_search as mgs
from ml_run_lib import tpot_run
#from ml_run_lib import dataset_analysis
#from ml_run_lib.dataset_analysis import pygam_classification

from organiser import repartition as r
'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.testing import all_estimators
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import ExtraTreeClassifier

from sklearn.ensemble import VotingClassifier

import xgboost

gaskl_model_catcher=0
n_splits=5
kf = KFold(n_splits=n_splits)

preprocessors=[\
        ('MinMaxAbs',MinMaxScaler()),\
        ('L2_norm',Normalizer(norm='l2')),\
        ('L1_norm',Normalizer(norm='l1'))]    

try :
  classifiers=[est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
  classifiers.append(('xgboost_classifier', xgboost.XGBClassifier))
except :
  classifiers=list()
  
'''

def make_an_exp_name_old(exp_name, exp_norm_file='experiment'):
    if not os.path.exists(exp_norm_file):
        os.makedirs(exp_norm_file)
    exp_file_name = exp_norm_file + '/' + exp_name
    if not os.path.exists(exp_file_name):
        os.makedirs(exp_file_name)
    return str(exp_file_name+'/'+exp_name)

def make_a_meta_exp_name(exp_norm_file='experiment', exp_name='dummy_experiment'):
#Output path will be of the format :
#exp_norm_file/exp_name[incrementation_if_necessary]
    if not os.path.exists(exp_norm_file):
        os.makedirs(exp_norm_file)
        
    if not os.path.exists(exp_norm_file + '/' + exp_name ):
        exp_file_name = exp_norm_file + '/' + exp_name 
        os.makedirs(exp_file_name)
        return os.path.abspath(exp_file_name), exp_name
    
    else :
        i=0
        while os.path.exists(exp_norm_file + '/' + exp_name + '_'+ str (i) ):
            i+=1
        os.makedirs(exp_norm_file + '/' + exp_name +'_'+ str (i) )
        exp_name=exp_name +'_'+ str (i)
        return str(os.path.abspath(exp_norm_file + '/' + exp_name )), exp_name

def make_a_ds_run_name_fp(exp_path, name='df_name', raise_if_exists=True):
#Output path will be of the format :
#exp_norm_file/exp_name[incrementation_if_necessary]/name/name
    if not os.path.exists(exp_path):
        raise
    
    if not os.path.exists(exp_path + '/exp_dirs/' + name + '/first_phase'):
        exp_file_name = exp_path + '/exp_dirs/' + name + '/first_phase'
        os.makedirs(exp_file_name)
        return exp_path + '/exp_dirs/' + name + '/first_phase/' +name 

    elif os.path.exists(exp_path + '/exp_dirs/' + name + '/first_phase') and raise_if_exists :
        raise
    
    else :
        return exp_path + '/exp_dirs/' + name + '/first_phase/' +name

def make_a_ds_run_name_2pb(exp_path, name='df_name',subdir='exp_dirs_2', raise_if_exists=True):
#Output path will be of the format :
#exp_norm_file/exp_name[incrementation_if_necessary]/name/name
    if not os.path.exists(exp_path):
        raise
    if not os.path.exists(exp_path + '/'+subdir+'/' + name + '/second_phase_b'):
        exp_file_name = exp_path + '/'+subdir+'/' + name + '/second_phase_b'
        os.makedirs(exp_file_name)
        return exp_path + '/'+subdir+'/' + name + '/second_phase_b/' +name 

    elif os.path.exists(exp_path + '/'+subdir+'/' + name + '/second_phase_b') and raise_if_exists :
        raise
        
    else :
        return exp_path + '/'+subdir+'/' + name + '/second_phase_b' + name


def produce_the_namer(exp_path, subdir='exp_dirs_2'):
    def namer(name): 
        return make_a_ds_run_name_2pb(exp_path, name,subdir)
    return namer

def make_an_exp_name(exp_name, exp_norm_file='experiment', name='df_name'):
#Output path will be of the format :
#exp_norm_file/exp_name_name[incrementation_if_necessary]/name/exp_name+'_'+name
    if not os.path.exists(exp_norm_file):
        os.makedirs(exp_norm_file)
    
    if not os.path.exists(exp_norm_file + '/' + exp_name + '/' + name):
        exp_file_name = exp_norm_file + '/' + exp_name + '/' + name
        os.makedirs(exp_file_name)
        return exp_file_name+'/'+ exp_name+'/'+ name+'/'+ exp_name+'_'+name

    else :
        i=0
        while os.path.exists(exp_norm_file + '/' + exp_name + str (i) + '/' + name):
            i+=1
#        if not os.path.exists(exp_norm_file + '/' + exp_name + str (i) + '/' + name):
        os.makedirs(exp_norm_file + '/' + exp_name + str (i) + '/' + name)

        return str(exp_norm_file + '/' + exp_name + str (i) + '/' + name+'/'+ exp_name+'_'+name)

def save_experiment_variables(df, exp_name):
    df.to_csv(str(exp_name+' features and target.csv'))
    second_df = df.copy()
    pd.concat([second_df, second_df.describe()],axis=0).to_csv(str(exp_name+' features and target with desc.csv'))

def save_main(exp_name):
    current_code = os.path.realpath(__file__)
    shutil.copy(current_code, str(exp_name+' main.py'))
"""
def classifiers_and_preprocessors_qt(df, target='ds', \
            classifiers=[], \
            preprocessors=[\
                           ('MinMaxAbs',MinMaxScaler()),\
                           ('L2_norm',Normalizer(norm='l2')),\
                           ('L1_norm',Normalizer(norm='l1'))]  \
        , name=None):
    if name is None : name=make_an_exp_name('cl_prepro_qt')
    for name_p, preprocessor in preprocessors :
        results_dfs= list()
        for train_index, test_index in kf.split(df.drop(target,axis=1).values):
            
            X_train, X_test = df.drop(target,axis=1).values[train_index], \
                df.drop(target,axis=1).values[test_index]
                
            y_train, y_test = df[target].values[train_index], \
                df[target].values[test_index]
            metrics=dict()
            
            for name_cl, clas in classifiers:
                try : 
                    print('----------')
                    print(name_cl)
                    model = clas()
                    hattori = Pipeline([('prepro', preprocessor), ('model', model)])
                    hattori.fit(X_train, y_train)
                    prediction = hattori.predict(X_test)
                    metrics[name_cl] = scorer.scoring_it(y_test, prediction, scorer.scoring)
                except ValueError:
                    print('Failed to train - ValueError')
                    pass
            results_dfs.append(pd.DataFrame(metrics).transpose())
        
        final_df,i =pd.DataFrame(), 0
        
        for df_chose in results_dfs[:-1]:
            if not(df_chose.empty) and final_df.empty:
                final_df=df_chose.copy()
                i+=1
            elif not(df_chose.empty) and not(final_df.empty):
                final_df+=df_chose
                i+=1
    
        #final_df = final_df.divide(i).sort_values(by='balanced accuracy',ascending=False) 
        final_df = final_df.divide(i)
        final_df.to_csv(name+name_p+'.csv')
        del final_df, results_dfs
        
    return 0
##        scorer.f1_scorer

def last_train_test(df_train, df_test, target, pipelines, name='cl_prepro_qt'):

    metrics_ft=dict()
#
    for name_p, pipeline in pipelines:
        try :
            print('----------')
            print(name_p)
            pipeline.fit(df_train.drop(target,axis=1), df_train[target])
            prediction = pipeline.predict(df_test.drop(target,axis=1))
            metrics_ft[name_p] = scorer.scoring_it(df_test[target], prediction, scorer.scoring)
        except ValueError:
            print('Failed to train - ValueError')
            pass
        
        pd.DataFrame(metrics_ft).transpose().to_csv(name+'_test_on_test.csv')
"""