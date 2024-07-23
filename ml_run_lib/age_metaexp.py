# -*- coding: utf-8 -*-
"""
@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from ml_run_lib.tpot_metaexperiment import TPOTMetaExperiment
from ml_run_lib.askl_metaexperiment import ASKLMetaExperiment
import defined_datasets.ds_normed as ds
from sklearn.metrics import balanced_accuracy_score, make_scorer, mean_squared_error, mean_absolute_error
#from ml_run_lib.custom_scores_aml import MAE_AML_score
from ml_run_lib.config_dict import regressor_config_dict
import autosklearn.metrics as am_metrics

db1 = [['Sex', bool],['Age', float]]

class AgeMetaExperiment(TPOTMetaExperiment):
    
    def __init__(self, dummy=False, name='added_name',exp_norm_fold='add_experiment_folder',rs=23, library=None, *args, **kwargs):
        #super().__init__(exp_norm_fold,name,*args, **kwargs)
        if dummy : experience_name='exp_age_dummy_'+name
        else : experience_name='exp_age_'+name
        
        super().__init__(exp_norm_fold,experience_name,rs=rs)
        
        if dummy : self.set_library(ds.spec_ds_lib)
        else : self.set_library(library)

        self.ds_libs_all_indices_to_same()
        self.init_train_test_indices_matching_splits_in_two(parameters=db1,rs=rs)
        self.init_2b_3b_indices_matching_splits_in_two(parameters=db1,rs=rs)
        
        #target should be set after matching 
        self.set_target(target='Age', classification=False)
        #Standard configuration
        if dummy:
          self.phase_1_exp_other_args={'threadp':4,'gen':2,'pop':50,'tpot_proc':1,'config_dict':regressor_config_dict}
          self.phase_2b_exp_other_args={'threadp':4,'gen':2,'pop':50,'tpot_proc':1,'config_dict':regressor_config_dict}
        else:
          self.phase_1_exp_other_args={'threadp':4,'gen':10,'pop':400,'tpot_proc':10,'config_dict':regressor_config_dict}
          self.phase_2b_exp_other_args={'threadp':4,'gen':10,'pop':400,'tpot_proc':10,'config_dict':regressor_config_dict}
        

        self.train_scorer = make_scorer(mean_absolute_error,greater_is_better=False)
        
        #self.phase_2b_exp_score = make_scorer(mean_squared_error,greater_is_better=False)
        self.phase_2b_exp_score = make_scorer(mean_absolute_error,greater_is_better=False)
        
        self.test_scorer = {'MSE':make_scorer(mean_squared_error,greater_is_better=False), 
                              'MAE':make_scorer(mean_absolute_error,greater_is_better=False),
                              #'SPMAE':MAE_AML_score
                              }
        self.test_before_exp()   

class AgeAsklExperiment(ASKLMetaExperiment):
    
    def __init__(self, dummy=False, name='default',exp_norm_fold='add_experiment_folder',rs=23, library=None, *args, **kwargs):
        #super().__init__(exp_norm_fold,name,*args, **kwargs)
        if dummy : experience_name='askl_age_dummy_'+name
        else : experience_name='askl_age_'+name
        
        super().__init__(exp_norm_fold,experience_name,rs=rs)
        if dummy : self.set_library(ds.spec_ds_lib)
        else : self.set_library(library)

        self.ds_libs_all_indices_to_same()
        self.init_train_test_indices_matching_splits_in_two(parameters=db1,rs=rs)
        self.init_2b_3b_indices_matching_splits_in_two(parameters=db1,rs=rs)
        
        #target should be set after matching 
        self.set_target(target='Age', classification=False)
        if dummy :
          self.phase_1_exp_other_args={'threadp':1, 'askl_2':True, 'askl_config':{'time_left_for_this_task':30, 'memory_limit': 1024*5,'n_jobs':10, \
          'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}
          self.phase_2b_exp_other_args={'threadp':1, 'askl_2':True, 'askl_config':{'time_left_for_this_task':30, 'memory_limit': 1024*1,'n_jobs':10, \
          'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}
        else:
          #Standard configuration
          self.phase_1_exp_other_args={'threadp':1, 'askl_2':True, 'askl_config':{'time_left_for_this_task':3600*24, 'memory_limit': 3072*4,'n_jobs':40, \
          'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}
          self.phase_2b_exp_other_args={'threadp':1, 'askl_2':True, 'askl_config':{'time_left_for_this_task':3600*24, 'memory_limit': 3072*4,'n_jobs':40, \
          'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}

        self.train_scorer = am_metrics.mean_squared_error
        
        self.phase_2b_exp_score = am_metrics.mean_squared_error
        
        self.test_scorer = {'MSE':mean_squared_error, 
                              'MAE':mean_absolute_error,
                              }
                              
        self.test_before_exp()   

