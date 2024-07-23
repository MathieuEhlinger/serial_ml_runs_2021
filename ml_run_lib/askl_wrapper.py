import numpy as np
from scipy import stats
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from copy import copy

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

class ASKLClassifier(ClassifierMixin,BaseEstimator):
  """
  After an auto-sklearn run ended, it has to be manipulated the same way an sklearn-model can be manipulated
  This class serves as a wrapper to allow this for regression
  """
  def __init__(self, core_model=None):
    self.core_model=core_model
    
  def fit(self, X, y):
    X, y = check_X_y(X, y)
    self.X_ = X
    self.y_ = y      
    return self
    
  def predict(self, X):
    check_is_fitted(self)
    X = check_array(X)
    return  self.core_model.predict(X)
  
  def get_params(self, deep=True):
    return {'core_model':self.core_model}
  
  def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      return self
    
class ASKLRegressor(RegressorMixin,BaseEstimator):
  """
  After an auto-sklearn run ended, it has to be manipulated the same way an sklearn-model can be manipulated
  This class serves as a wrapper to allow this for regression
  """
  def __init__(self, core_model=AutoSklearnRegressor(time_left_for_this_task = 30)):
    self.core_model=core_model
    
  def fit(self, X, y):
    try : 
      self.core_model.refit(X,y)
    except : 
      askl = AutoSklearnClassifier(time_left_for_this_task = 30)
      askl.fit(X,y)
      self.core_model=askl
      self.core_model.refit(X,y)
    X, y = check_X_y(X, y)
    self.X_ = X
    self.y_ = y
    return self
    
  def predict(self, X):
    check_is_fitted(self)
    X = check_array(X)
    return  self.core_model.predict(X)
  
  def get_params(self, deep=True):
    return {'core_model':self.core_model}
  
  def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      return self