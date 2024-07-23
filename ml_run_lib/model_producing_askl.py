# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:48:54 2019

@author: Mathieu Ehlinger
Part of the MetaExperiment Project

"""

import pandas as pd
from numpy.random import seed


from multiprocessing import Pool as ThreadPool
#from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
#multiprocessing.set_start_method('forkserver')

import subprocess

from ml_run_lib.experiments_functions_new import make_a_ds_run_name_fp

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, make_scorer
import itertools
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

from autosklearn.experimental.askl2 import AutoSklearn2Classifier
#from autosklearn.experimental.askl2 import AutoSklearn2Regressor

import pickle
import copy
import os
from shutil import copyfile
from askl_wrapper import ASKLClassifier, ASKLRegressor

def train_askl_ft(features,targets, name, scorer, save = True,
               target='ds', classification=True,
               askl_config=None, rand=23,verbosity=1,save_entry=True,askl_2=True):
               
               
    if save_entry : features.to_csv(path_or_buf=(name + '_features.csv'), index = True)
    if save_entry : targets.to_csv(path_or_buf=(name + '_targets.csv'), index = True)

    print('---------------------')

    #if save :
    #    pd.DataFrame(features).to_csv(path_or_buf=(name + ' features.csv'), index = True)
    #    pd.DataFrame(targets).to_csv(path_or_buf=(name + ' targets.csv'), index = True)



#    features = features.values  #Making sure the index names have no influence
#    target = target.values

#    X_train, X_test, y_train, y_test = train_test_split(
#            features,target,
#            train_size=0.75, test_size=0.25)
    print('Scorer :', scorer)
    if classification :
        if askl_2:
          print('ASKL-Classifier 2 search running ...')
          askl = AutoSklearn2Classifier( **askl_config,
                                     seed=rand,
                                     tmp_folder=name+'.temp',
                                     delete_tmp_folder_after_terminate=False,
                                     metric=scorer
                                     )
        else:
          print('ASKL-Classifier search running ...')
          askl = AutoSklearnClassifier( **askl_config,
                                     seed=rand,
                                     tmp_folder=name+'.temp',
                                     delete_tmp_folder_after_terminate=False,
                                     metric=scorer
                                     )
    else :
          print('ASKL-Regressor search running ...')
          askl = AutoSklearnRegressor(**askl_config,
                                     seed=rand,
                                     tmp_folder=name+'.temp',
                                     delete_tmp_folder_after_terminate=True,
                                     metric=scorer
                                     )

    askl.fit(features, targets)
    
    if classification :
      askl = ASKLClassifier(core_model=askl)
    else:
      askl = ASKLRegressor(core_model=askl)
    
    with open((name+'_model.pickle'), 'wb') as pickle_file:
        pickle.dump(askl, pickle_file)
    results = pd.DataFrame(askl.core_model.cv_results_)
    results.to_csv(name+'_run_results.csv')
    if save :
        with open((name+'_train_var.pickle'), 'wb') as pickle_file:
            pickle.dump({'features': features,'targets':targets,
                         'function' : scorer},
                         pickle_file)

    return askl

def fp_for_mt_askl(df_o,target_ds, exp_path='experiment', target='ds',rand=23,classification=True,scorer=None, askl_config=None, naming_func=make_a_ds_run_name_fp, askl_2=False):
    #first phase experiment function, designed for multithreading
    #df is from the DataSet class here
    df = df_o.df_data
    seed(rand)
    result=dict()
    print('Experience name :',df_o.ds_name)
    #X_train, X_test, y_train, y_test = train_test_split(df, target_ds, test_size=0.33, random_state=rand)
    #print('Naming function :',naming_func)
    name = naming_func(exp_path, df_o.ds_name)

    #model, fitted=train_askl_ft(features=df,targets=target_ds, gen=gen, pop=pop,
    return df_o.ds_name, train_askl_ft(features=df,targets=target_ds,
        name=name, scorer=scorer, save = True,
        target=target, classification=classification,
        askl_config=askl_config, rand=rand, askl_2=askl_2)
        
    '''
    features,targets, name, scorer, save = True,
               target='ds', classification=True,
               config_dict=None, rand=23,verbosity=1,save_entry=True
    '''
    #result = fitted.score(X_test, y_test)
    #pd.DataFrame.from_dict(result).to_csv(str(name+'_askl_runs_results.csv'))
    #return [df_o.ds_name, model]
    

def multithreading_askl(ds_library, target_ds, exp_path='experiment', target='ds', classification=True,scorer=None,rand=23,arg_dict={},naming_func=make_a_ds_run_name_fp):
    #Reception of the arg_dict is done here
    #print('Arg_dict :',arg_dict)
    if 'threadp' in arg_dict.keys() : threadp = arg_dict['threadp']
    else : threadp = 1
    if 'askl_2' in arg_dict.keys() : askl_2 = arg_dict['askl_2']
    else : askl_2 = False
    if 'askl_config' in arg_dict.keys() : askl_config = arg_dict['askl_config'] 
    else : askl_config={'time_left_for_this_task':1200}
    #Some chatting
    print('---------------------\n')
    print('Auto-skl experiment initiated ! Running on ', str(len(ds_library.keys())),' libraries.') 
    pool = ThreadPool(threadp)
    #Lets go !
    #print('Naming function :',naming_func)
    print('Configuration :',askl_config)
    print('\n---------------------')
    '''
    models = pool.starmap(fp_for_mt_askl, zip(list(ds_library.values()), itertools.repeat(target_ds),itertools.repeat(exp_path),\
      itertools.repeat(target),itertools.repeat(rand), itertools.repeat(gen),\
      itertools.repeat(classification), itertools.repeat(scorer), itertools.repeat(askl_proc), itertools.repeat(pop), 
      itertools.repeat(naming_func),itertools.repeat(config_dict) ))
    '''
    models = pool.starmap(fp_for_mt_askl, zip(list(ds_library.values()), itertools.repeat(target_ds),itertools.repeat(exp_path),\
      itertools.repeat(target),itertools.repeat(rand), \
      itertools.repeat(classification), itertools.repeat(scorer), itertools.repeat(askl_config),
      itertools.repeat(naming_func),itertools.repeat(askl_2)))
    end_dict=dict()
    for key, model in models :
      end_dict[key]=model
      
    return end_dict
    
def multiproc_askl(ds_library, target_ds, exp_path='experiment', target='ds', classification=True,scorer=None,rand=23,arg_dict={},naming_func=make_a_ds_run_name_fp):
    #Reception of the arg_dict is done here
    #print('Arg_dict :',arg_dict)
    if 'threadp' in arg_dict.keys() : threadp = arg_dict['threadp']
    else : threadp = 1
    if 'askl_2' in arg_dict.keys() : askl_2 = arg_dict['askl_2']
    else : askl_2 = False
    if 'askl_config' in arg_dict.keys() : askl_config = arg_dict['askl_config'] 
    else : askl_config={'time_left_for_this_task':1200}
    #Some chatting
    print('---------------------\n')
    print('Auto-skl experiment initiated ! Running on ', str(len(ds_library.keys())),' libraries.') 
    pool = ThreadPool(threadp)
    #Lets go !
    #print('Naming function :',naming_func)
    print('Configuration :',askl_config)
    print('\n---------------------')
    models=list()
    for lib_name in ds_library: 
      fp_for_mt_askl(ds_library[lib_name], target_ds,exp_path,\
        target,rand, \
        classification, scorer, askl_config,
        naming_func,askl_2)
    end_dict=dict()
    for key, model in models :
      end_dict[key]=model
      
    return end_dict