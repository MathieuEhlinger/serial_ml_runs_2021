import sys

import pathlib
import itertools
import tpot
import pickle
import copy
import os
import pandas as pd

from tpot import TPOTClassifier
from tpot import TPOTRegressor

# -*- coding: utf-8 -*-
"""
Created on ?

@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""

import sys, os
#sys.path.append('./../datai')

#get_cwd = os.getcwd()
#os.chdir(os.path.dirname(__file__))

import numpy as np
import ml_run_lib.modelslists as modelslists
import pandas as pd
import os
from ml_run_lib.transformers import nichts

from datai.scorer import roc_auc_scorer
from datai.scorer import kappa

from numpy.random import seed
import sklearn
from sklearn import naive_bayes, linear_model
from sklearn import tree, svm, preprocessing
from sklearn import ensemble, neighbors, discriminant_analysis

from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score, log_loss,roc_auc_score, balanced_accuracy_score
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score

def watch_class_and_params(class_param_dict):
    for class_name in class_param_dict.keys() :
        parameters = class_param_dict[class_name]
        exec(str('model='+class_name+'()'))
        print('-----------')
        print(model, '\n')
        print(parameters)

def load_models(class_param_dict):
    ignored_keys=[]
    for class_name in class_param_dict.keys() :
        try :
            param_grid = class_param_dict[class_name]
            model = eval(str(class_name+'()'))
        except :
            print(class_name,' could not be loaded. Entry ignored')
            ignored_keys.append(class_name)
    
    for class_name in ignored_keys:
        class_param_dict.pop(class_name)
    return class_param_dict

def massive_grid_search(df, target, class_param_dict, 
                        scoring = make_scorer(balanced_accuracy_score), cv=5):
    """
    Core of the Grid-searching.
    1. Loads models using load_models (checks models availability)
    2. Copies dataset
    3. Loops through the {'model': params_grid}
        - Model loads
        - Attempts grid search
        - with cv, results in the form of {'model_name':{results_for_classifier}}
    4. Returns the results
    
    """
    class_param_dict=load_models(class_param_dict)
    models = dict()
    X_train, y_train = df.drop(target, axis=1).copy(),df[target].copy()
    results = dict()
    for class_name in class_param_dict.keys() :
        param_grid = class_param_dict[class_name]
        model = eval(str(class_name+'()'))
        print('-----------')
        print(model, '\n')
        print(param_grid)
        try :
            grid = GridSearchCV(estimator=model,param_grid=param_grid,
                            refit = True, verbose=0, scoring =scoring, cv=cv)
            grid.fit(X_train,y_train)
            results[class_name]=[grid.best_score_, grid.best_params_]
            models[class_name] = clone(model)
        except AttributeError: 
                print(class_name,'attribute error - ignored')
        
    return results, models

def result_exploration(df_o, target, class_list_o, cv=5, rand=23, \
                       scoring = None):
    """
    A nasty dictionnary of results came out. Time to make it a little more beautiful
    
    """
    if scoring == None :
        scoring={\
            
            'accuracy':accuracy_score,
            'balanced_accuracy_score':balanced_accuracy_score,
            'precision':    lambda x,y :precision_score(x,y, average='macro'),
            'recall':       lambda x,y :recall_score(x,y, average='macro'),
            'auc':roc_auc_scorer
            }
        
    df, class_list = df_o.copy(),class_list_o.copy()
    results=dict()
    for key in scoring.keys():
        x = make_scorer(scoring[key])
        scoring[key]= x    
        
    for name in list(class_list.keys()):
        scores=dict()
        clas = eval(str(name+'()'))
        clas.get_params(class_list[name][1])
        seed(rand)
#        print(name,':',clas)
        scores_o = cross_validate(clas,  df.drop(target, axis =1),\
                                    df[target], cv=cv, scoring = scoring)
        for key in scores_o.keys():
            scores[str('mean_'+key)]=np.round(np.mean(scores_o[key]),3)
            scores[str('mean_std_'+key)]=str(np.round(np.mean(scores_o[key]),3))\
            +' +/- '+str(np.round(np.std(scores_o[key]),3))
        scores['best_test_config']=class_list[name][1]
        results[name]=scores
            
#        except TypeError :
#            print(name,' doesn\'t have fit method')
#        except Exception as e: print(e)
#    print('Results :',results)
    print(clas)
    return results

def find_keys(one_dict, chain='test'):
    bad_keys=[]
    for key in one_dict.keys():
        if chain not in key :
            bad_keys.append(key)
    return bad_keys

def saving_ranks_results(result_dict_o, name='test', \
                         dict_filter='test', fil=True):
    result_dict = result_dict_o.copy()
    if fil : bad_keys = find_keys(result_dict[list(result_dict.keys())[0]])
    result_df = pd.DataFrame(result_dict).transpose()
    if fil : result_df = result_df.drop(bad_keys, axis=1)
    result_df=result_df.sort_values('mean_test_balanced_accuracy_score',ascending=False)
#    except: 
#        print('Sorting failed')
#        raise
    result_df.to_csv(name+'_GS_with_CV.csv')

def grid_searching_through_the_night(df, target_df, name,target='ds',rand=23,classification=True,
                                     scoring= make_scorer(balanced_accuracy_score),
                                     predictors_dict=standard_classifiers,
                                     dict_filter='test', 
                                     fil=False,cv=3):
    '''
    1. Grid searches through the classes, returns a dict of results
        - massive_grid_search uses only 1 scorer. By default : CKS
    2. Result exploration to make it more understandable
        - If scoring=None is fed :
        'accuracy','precision', 'recall','auc' are def.
    
    '''
    seed(rand)
    result=dict()
    print('Experience path :',name,'\n - Number of classifiers tested :', len(predictors_dict))

    results, models = massive_grid_search(df, target_df, target, predictors_dict,cv=cv)
    results2 = result_exploration(df,target_df,target, results, cv=cv, rand=rand, scoring=scoring)
    saving_ranks_results(results2, name=name, dict_filter=dict_filter, fil=fil)


if __name__ == '__main__' :
  exp_path  = str(os.path.dirname(os.path.abspath(__file__)))
  exp_name  = str(exp_path.split('/')[-2])
  name = exp_path+'/'+exp_name
  features  = pd.DataFrame.from_csv(name+'_features.csv')
  targets   = pd.DataFrame.from_csv(name+'_targets.csv')
  
  with open((name+'_main_lib_version.txt'), 'w') as infile :
      infile.write('Scikit-learn version : ' + str(sklearn.__version__))
  
  with open(name + '_configs.pickle', "rb" ) as pickle_file:
      pic = pickle.load( pickle_file )
      '''
      {'gen':gen, 'pop':pop, 'name':name, 'scorer':scorer, 'save':save,
                 'proc':proc,'target' :target, 'classification':classification,
                 'config_dict':config_dict, 'rand':rand}, open( name + '_configs.pickle', "wb" )
      '''
      scorer         = pic['scorer']
      classification = pic['classification']
      n_proc         = pic['proc']
      rand           = pic['rand']
      config_dict    = pic['config_dict']
      target         = pic['target']
      cv             = pic['cv']
  
  grid_searching_through_the_night(df=features,target_df=targets, name=name,scoring=scorer,\
        classification=classification, proc=n_proc, save=True,save_entry=False ,predictors_dict=config_dict,rand=rand,target=target,cv=cv)