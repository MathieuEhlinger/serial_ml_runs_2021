# -*- coding: utf-8 -*-
"""
@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""
from ml_run_lib.tpot_metaexperiment import TPOTMetaExperiment
import defined_datasets.new_datasets as ds
from sklearn.metrics import balanced_accuracy_score, precision_score,  recall_score, f1_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from ml_run_lib.config_dict import classifier_config_dict

db1 = [['Age', float], ['Gender', bool]]    

class MDDMetaExperiment(TPOTMetaExperiment):
    
    def __init__(self, dummy=False, name='added_name',exp_norm_fold='add_experiment_folder',rs=23, library=ds.spec_ds_lib,*args, **kwargs):
        #super().__init__(exp_norm_fold,name,*args, **kwargs)
        if dummy : experience_name='exp_MDD_dummy_'+name
        else : experience_name='exp_MDD_'+name
        
        super().__init__(exp_norm_fold,experience_name,rs=rs)
        
        if dummy : self.set_library(ds.test_ds_lib)
        else : self.set_library(library)

        self.ds_libs_all_indices_to_same()
        self.init_train_test_indices_matching_splits_in_two(parameters=db1,rs=rs)
        self.init_2b_3b_indices_matching_splits_in_two(parameters=db1,rs=rs)
        
        #target should be set after matching 
        self.set_target(target='MDD', classification=True)
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