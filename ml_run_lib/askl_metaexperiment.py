# -*- coding: utf-8 -*-
"""
@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""
from ml_run_lib.metaexperiment import MetaExperiment
import os 
import numpy as np
import pandas as pd
import pickle
import ml_run_lib.model_producing_askl 
from ml_run_lib.experiments_functions_new import make_a_meta_exp_name, make_a_ds_run_name_fp
from sklearn.model_selection import KFold
from sklearn.utils import indexable
from copy import deepcopy
#from ml_run_lib.askl_code_reader import askl_code_execution, askl_model_score

class ExperimentSaveNotFoundError(Exception):
    pass

def cross_validate_askl(estimator, X, y, scoring=None,cv=10,rs=23):
      #cross_validate_askl(models[dataset], X, y,scoring=self.test_scorer, cv=10)
      #    
      X, y, groups = indexable(X, y, None)
      kf= KFold(n_splits=cv,shuffle=True, random_state=rs)
      kf.get_n_splits(X) 
      scores=[]
      splits_index=[]
      i=0
      for train_index, test_index in kf.split(X):
          i+=1
          X_train, X_test = X.iloc[train_index], X.iloc[test_index]
          y_train, y_test = y.iloc[train_index], y.iloc[test_index]
          splits_index.append({'Train_index_'+str(i):deepcopy(train_index), 'Test_index_'+str(i): deepcopy(test_index)})
          estimator.fit(X_train, y_train)
          y_pred=estimator.predict(X_test)
          if type(scoring)==type(dict()):
            scores_dict={}
            for sc in scoring:
              scores_dict[sc]=scoring[sc](y_test,y_pred)
            scores.append(scores_dict)
          else:
            scores.append({str(scoring):scoring(y_test,y_pred)})
      
      final_dict=dict()
      for key in scores[0]:
        temp_list=list()
        for result in scores:
          temp_list.append(result[key])
        final_dict['test_'+key]=deepcopy(temp_list)
      final_dict['fit_time']=deepcopy(temp_list)
      final_dict['score_time']=[None for i in range(0,cv)]
      return final_dict, {k:v for x in splits_index for k,v in x.items()}

class ASKLMetaExperiment(MetaExperiment):
    #--------------------------------------------------------------------------------
    #General manipulation methods for the models_dict attribute, specifically related to askl experiments
    #Modified to return a dict of complete models and a list of uncomplete models, allowing usage in phase 2 b
    def __init__(self, exp_norm_fold='add_experiment_folder', experience_name='dummy_exp',rs=23):
        super().__init__(exp_norm_fold=exp_norm_fold, experience_name=experience_name,rs=rs)
        #self.phase_1_exp             = ml_run_lib.model_producing_askl.multithreading_askl
        self.phase_1_exp             = ml_run_lib.model_producing_askl.multiproc_askl
        #self.phase_2b_exp_model_prod = ml_run_lib.model_producing_askl.multithreading_askl
        self.phase_2b_exp_model_prod = ml_run_lib.model_producing_askl.multiproc_askl
        self.cross_validate= cross_validate_askl
    
    def salvage_gen_models_askl(self, subdir='exp_dirs',phase='first_phase'):
      #Lets recover models from tpot runs
      #return list of libs name for which mature model could not be found
      temp_model_dict = dict()
      if not os.path.exists(self.meta_experience_path+'/'+subdir):  raise ExperimentSaveNotFoundError('Experiment does not exist or wasn\'t saved')
      
      all_dirs=os.listdir(self.meta_experience_path+'/'+subdir)
      uncomplete_models=list()
      for exp in all_dirs :
          print('\nLoading experiment models : ',exp,end='')
          if not os.path.isfile(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+phase+'/'+exp+'_model.pickle'):
                print(' - WARNING : FAILURE Model in early stage of generation - Can not be loaded.', end='')
                uncomplete_models.append(exp)
                temp_model_dict[exp]= None
          else : 
              with open(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+phase+'/'+exp+'_model.pickle','rb') as infile :
                  results = pickle.load( infile )
                  temp_model_dict[exp]=deepcopy(results)
              print(' - SUCCESS : Model Loaded from pickle', end='')
      print('')
      print('Unfinished models are considered to be (',len(uncomplete_models),'):',uncomplete_models)
      return temp_model_dict, uncomplete_models
    

    