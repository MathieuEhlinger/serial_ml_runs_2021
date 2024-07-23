# -*- coding: utf-8 -*-
"""
@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""
from ml_run_lib.tpot_metaexperiment import TPOTMetaExperiment
import defined_datasets.new_datasets as ds
from sklearn.metrics import balanced_accuracy_score, make_scorer, mean_squared_error, mean_absolute_error
#from ml_run_lib.custom_scores_aml import MAE_AML_score
from ml_run_lib.config_dict import regressor_config_dict

db1 = [['eTIV_spm', float], ['Gender', bool]]    

class ETIVMetaExperiment(TPOTMetaExperiment):
    
    def __init__(self, dummy=False, name='added_name',exp_norm_fold='add_experiment_folder',rs=23, library=None, *args, **kwargs):
        #super().__init__(exp_norm_fold,name,*args, **kwargs)
        if dummy : experience_name='exp_etiv_dummy_'+name
        else : experience_name='exp_etiv_'+name
        
        super().__init__(exp_norm_fold,experience_name,rs=rs)
        
        if dummy : self.set_library(ds.test_ds_lib)
        else : self.set_library(library)

        self.ds_libs_all_indices_to_same()
        self.init_train_test_indices_matching_splits_in_two(parameters=db1,rs=rs)
        self.init_2b_3b_indices_matching_splits_in_two(parameters=db1,rs=rs)
        
        #target should be set after matching 
        self.set_target(target='eTIV_spm', classification=False)
        self.phase_1_exp_other_args={'threadp':1,'gen':10,'pop':400,'tpot_proc':10,'config_dict':regressor_config_dict}
        self.phase_2b_exp_other_args={'threadp':1,'gen':10,'pop':400,'tpot_proc':10,'config_dict':regressor_config_dict}
        
        #self.train_scorer = make_scorer(mean_squared_error,greater_is_better=False)
        self.train_scorer = make_scorer(mean_absolute_error,greater_is_better=False)
        
        #self.phase_2b_exp_score = make_scorer(mean_squared_error,greater_is_better=False)
        self.phase_2b_exp_score = make_scorer(mean_absolute_error,greater_is_better=False)
        
        self.test_scorer = {'MSE':make_scorer(mean_squared_error,greater_is_better=False), 
                              'MAE':make_scorer(mean_absolute_error,greater_is_better=False),
                              #'SPMAE':MAE_AML_score
                              }
        self.test_before_exp()   

