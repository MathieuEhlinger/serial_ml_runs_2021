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
from sklearn.metrics import balanced_accuracy_score, precision_score,  recall_score, f1_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from ml_run_lib.config_dict import classifier_config_dict
import autosklearn.metrics as am_metrics

db1 = [['Sex', bool],['Age', float]]
db_tpot = [['Sex', bool],['Age', float]]       

class GenderMetaExperiment(TPOTMetaExperiment):
    
    def __init__(self, dummy=False, name='added_name',exp_norm_fold='add_experiment_folder',rs=23, library=ds.spec_ds_lib,*args, **kwargs):
        #super().__init__(exp_norm_fold,name,*args, **kwargs)
        if dummy : experience_name='exp_gender_dummy_'+name
        else : experience_name='exp_gender_'+name
        
        super().__init__(exp_norm_fold,experience_name,rs=rs)
        
        if dummy : self.set_library(ds.spec_ds_lib)
        else : self.set_library(library)

        self.ds_libs_all_indices_to_same()
        self.init_train_test_indices_matching_splits_in_two(parameters=db1,rs=rs)
        self.init_2b_3b_indices_matching_splits_in_two(parameters=db1,rs=rs)
        
        #target should be set after matching 
        self.set_target(target='Gender', classification=True)
        self.phase_1_exp_other_args={'threadp':12,'gen':5,'pop':300,'tpot_proc':20,'config_dict':classifier_config_dict}
        self.phase_2b_exp_other_args={'threadp':12,'gen':5,'pop':300,'tpot_proc':20,'config_dict':classifier_config_dict}
        
        self.train_scorer = make_scorer(balanced_accuracy_score,greater_is_better=True)
        
        self.phase_2b_exp_score = make_scorer(balanced_accuracy_score,greater_is_better=True)
        
        self.test_scorer = {'balanced_accuracy_score':make_scorer(balanced_accuracy_score,greater_is_better=True), 
                              'precision':make_scorer(precision_score,greater_is_better=True),
                              'recall':make_scorer(recall_score,greater_is_better=True),
                              'f1_score':make_scorer(f1_score,greater_is_better=True)
                              }
        
        self.test_before_exp()
        
class GenderAsklExperiment(ASKLMetaExperiment):
    
    def __init__(self, dummy=False, name='default',exp_norm_fold='add_experiment_folder',rs=23, library=None, *args, **kwargs):
        #super().__init__(exp_norm_fold,name,*args, **kwargs)
        if dummy : experience_name='askl_gender_dummy_'+name
        else : experience_name='askl_gender_'+name
        
        super().__init__(exp_norm_fold,experience_name,rs=rs)
        if dummy : self.set_library(ds.spec_ds_lib)
        else : self.set_library(library)

        self.ds_libs_all_indices_to_same()
        self.init_train_test_indices_matching_splits_in_two(parameters=db1,rs=rs)
        self.init_2b_3b_indices_matching_splits_in_two(parameters=db1,rs=rs)
        
        #target should be set after matching 
        self.set_target(target='Sex', classification=True)
        if dummy :
          self.phase_1_exp_other_args={'threadp':1, 'askl_2':True, 'askl_config':{'time_left_for_this_task':30, 'memory_limit': 3072*4,'n_jobs':10, \
          'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}
          self.phase_2b_exp_other_args={'threadp':1, 'askl_2':True, 'askl_config':{'time_left_for_this_task':30, 'memory_limit': 3072*4,'n_jobs':10, \
          'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}
        else:
          #Standard configuration
          self.phase_1_exp_other_args={'threadp':1, 'askl_2':True, 'askl_config':{'time_left_for_this_task':3600*24, 'memory_limit': 3072*4,'n_jobs':40,}} 
#          'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}
          self.phase_2b_exp_other_args={'threadp':1, 'askl_2':True, 'askl_config':{'time_left_for_this_task':3600*24, 'memory_limit': 3072*4,'n_jobs':40, \
          'resampling_strategy':'cv', 'resampling_strategy_arguments':{'folds':10}}}

        self.train_scorer = am_metrics.balanced_accuracy
        
        self.phase_2b_exp_score = am_metrics.balanced_accuracy
        
        self.test_scorer = {'balanced_accuracy_score':balanced_accuracy_score, 
                              'precision':precision_score,
                              'recall':recall_score,
                              'f1_score':f1_score,
                              }
        self.test_before_exp()   
