# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:59:52 2018

@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""

import sys, os
sys.path.append('./../datai')
sys.path.append('./../pablo')

get_cwd = os.getcwd()
os.chdir(os.path.dirname(__file__))

import numpy as np
import pickle
import pandas as pd
import pytest
import ml_run_lib.transformers

from datai.scorer import f1_scorer
from datai.scorer import kappa
from datai.scorer import roc_auc_scorer

import time

from ml_run_lib.transformers import nichts

from sklearn.base import clone

from ml_run_lib.experiment_functions import assign_filename
from ml_run_lib.experiment_functions import pipeline_eval
from ml_run_lib.experiment_functions import pipeline_maker
from ml_run_lib.experiment_functions import make_an_excel

#from pablo.display_functions import pipeline_results_display
#from pablo.display_functions import pipeline_retest_results_display

from numpy.random import seed
from sklearn.model_selection import cross_val_score

from sklearn.utils.testing import all_estimators
from sklearn.base import ClassifierMixin

#from sklearn.metrics import accuracy_score, log_loss,auc, balanced_accuracy_score
from sklearn.metrics import accuracy_score, log_loss,auc
from sklearn.metrics import f1_score, cohen_kappa_score, make_scorer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#---------------------------
    
def all_classifier_search(df, file_name='test', scoring = None, 
                 rand=23, target='ds', cv=5):
    """
    1. 
    2. Saves experiment variables
    3. Builds the scorer from metric functions
    4. Takes a list of all classifiers available in scikit
    5. Test them with cross-validation
    6. Returns a dict composed from 'classifier':result_dict
    """

    z = locals()
    pd.Series(z).to_csv(path=(file_name+' experiment variables.csv'))

    if scoring == None :
        
                scoring_c={'log_loss': log_loss, 
                 'accuracy':accuracy_score,
#            'balanced_accuracy_score':balanced_accuracy_score,
            'precision':    lambda x,y :precision_score(x,y, average='macro'),
            'recall':       lambda x,y :recall_score(x,y, average='macro'),
            'auc':roc_auc_scorer}
    else : scoring_c=scoring.copy()
    seed(rand)
#    print(file_name)
#    with open(file_name+'locals.pickle', 'wb') as handle:
#        pickle.dump(z, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(str(scoring_c['accuracy']))
    
    for key in scoring_c.keys():
        x = make_scorer(scoring_c[key])
        scoring_c[key]= x
    

    df.to_csv(path_or_buf=(file_name+' dataset.csv'))
    classifiers=[est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
    results=dict()
    for name, clas in classifiers:
#        scores = cross_validate(clas(),  df.drop(target, axis =1),\
#                                    df[target], cv=cv, scoring = scoring_c)
#        results[name]=scores
#        print(name,':',clas)
        try :
            scores = cross_validate(clas(),  df.drop(target, axis =1),\
                                    df[target], cv=cv, scoring = scoring_c)
            results[name]=scores
        except TypeError :
            print(name,' doesn\'t have fit method')
        except Exception as e: print(e)
    return results


def check_all_classifiers(df, file_name='test',rand=23, 
                          scoring={'accuracy':accuracy_score,
                                   'cohen kappa score':kappa,
                                   'ROC AUC':auc,
#                                   'f1 score':f1_score,
#                                   'balanced accuracy':balanced_accuracy_score,
                                   'log loss':log_loss} , trans_f=[[nichts,[]]],
                                   cv=5, save=True, target='ds'):
    """
    1. Throw all parameters in all_classifier_search, gets the class/res in result
    2. Averages the values
    """
    
    r = all_classifier_search(df.copy(),rand=rand, scoring = scoring, 
                               file_name=file_name, cv=cv, target=target
                             )
    
    this, end_keys = dict(),list()
    for model in r.keys():
        this[model]=dict()
        for values in r[model].keys():
            this[model][values]=np.mean(r[model][values])
            
    result = pd.DataFrame.from_dict(this).transpose()
    try :
        result = result.sort_values(by='test_balanced accuracy',ascending=False)
        pass
    except :
        print('sorting failed')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    for key in result.keys():
        if 'test' in key:
            end_keys.append(key)
    result = result[end_keys]
    if save : result.to_csv((file_name+' - Results of traditionnal ML - '+str(timestr)+'.csv'))
    print('-------------\n')
    print('-------------')
    return result.index

os.chdir(get_cwd)