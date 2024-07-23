# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:18:36 2019

Je suis matinal, mas j'ai mal

@author: Mathieu
"""

#from sklearn import some_folding_function_sklearn
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
import copy
import pandas as pd
from datai.ds_building_functions import DSLibrary, DataSet

#some_folding_function = some_folding_function_sklearn

#def phase_4b_retest_sklearn(MetaExperiment, folding_function=some_folding_function,rs=23):
'''
    Tatintintin
    First step : generate 10 folds
    Folds have the form :   - 50% to train base predictors,
                            - 40% train the voting entity
                            - 10% predict with the voting entity

    Folds will be dicts ? Maybe Sklearn compatible

    Second step : Iterate over the folds. Produce scores
                            - Best practice : Make folds sklearn compatible

    Third step : Return meta voter

'''

def generate_exp_indices(X, raw_indices=None, refined_indices=None):
    '''
    Used to split between raw-model train set
    and refined-model train set
    '''
    if raw_indices is not None and refined_indices is not None :
        return raw_indices, refined_indices
    X_index = X[list(X.keys())[0]].df_data.index

    if raw_indices is not None and refined_indices is None :
        refined_indices = pd.Intersection.difference(X_index, raw_indices)
        return raw_indices, refined_indices

    if raw_indices is None and refined_indices is not None :
        raw_indices = pd.Intersection.difference(X_index, refined_indices)
        return raw_indices, refined_indices

    if raw_indices is None and refined_indices is None :
        len_of_index    = len(X_index)
        raw_indices_to  = int(len_of_index*0.6)
        raw_indices     = X_index[:raw_indices_to]
        refined_indices = pd.Index.difference(X_index, raw_indices)
        return raw_indices, refined_indices

class SuperPredictor(BaseEstimator):
    
    def update(self, raw_model_dict, over_predictor):
        #add a line to reset each model
        self.models_dict_raw     = copy.deepcopy(raw_model_dict)
        self.prediction_pipeline = clone(over_predictor)
    
    def __init__(self, raw_model_dict, over_predictor):
        self.update(raw_model_dict, over_predictor)
    
    def reset_all_models(self):
        for model in self.models_dict_raw :
          self.models_dict_raw[model] = clone(self.models_dict_raw[model]) 
          
    def train_all_raw_models(self,X,y,raw_indices):
        for model in self.models_dict_raw :
          self.models_dict_raw[model].fit(X[model].df_data.loc[raw_indices],y.loc[raw_indices])
    
    def train_refined_model(self, X_refined, y, refined_indices=None):
        
        self.prediction_pipeline.fit(X_refined.loc[refined_indices], y.loc[refined_indices])
    
    def generate_refined_data(self,X,refined_indices):
        X_refined = dict()
        for model in self.models_dict_raw :
          X_refined[model] = self.models_dict_raw[model].predict(X[model].df_data.loc[refined_indices])
        return pd.DataFrame(X_refined, index = refined_indices)  
    
    def fit(self,X,y=None, raw_indices=None,refined_indices=None):
    
        if y is None : y = X.target_ds
        self.reset_all_models()
        
        raw_indices,refined_indices = generate_exp_indices\
            (X,raw_indices=raw_indices,refined_indices=refined_indices)

        self.train_all_raw_models(X,y,raw_indices)

        X_refined = self.generate_refined_data(X,refined_indices)
        self.prediction_pipeline.fit(X_refined.loc[refined_indices], y.loc[refined_indices])
    
    def predict(self,X):
        X_refined = self.generate_refined_data(X, X[list(X.keys())[0]].df_data.index)
        y_pred = self.prediction_pipeline.predict(X_refined)
        return y_pred