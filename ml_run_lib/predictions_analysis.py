import datai.new_datasets as ds
from sklearn.model_selection import cross_val_predict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import sklearn.base
import seaborn as sns
import os


def save_without_deleting(self, dir_path, file_name = '_my_picture', extension = '.png'):
  # Just for not overwriting things. Also checks if dir exists.
  if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
  if dir_path is None : dir_path = self.meta_experience_path
   
  if not os.path.exists(dir_path+'/'+ self.meta_experience_name + file_name +'_0' + extension):
        save_path = dir_path+'/'+ self.meta_experience_name + file_name +'_0'+ extension
  else:
    allfiles = [f for f in listdir(dir_path) if isfile(join(dir_path+'/', f))]
    number = max([int(f.split('_')[-1][:-len(extension)]) for f in allfiles if ((self.meta_experience_name + file_name) in f and extension in f)])+1
    save_path = dir_path+'/'+ self.meta_experience_name + file_name +'_'+str(number)+extension
  return save_path

def produce_predictions_10_10_10F(X,y,model, rs=23, error=False):  
  #X and y are DataFrames
  #error = True f you want the error and not the predictions
  all_predictions=list()
  for i in range(10):
    np.random.seed(rs+i) 
    X_copy = X.sample(frac=1, replace=False, random_state=rs+i)
    y_copy = y.loc[X_copy.index]
    
    results = cross_val_predict(model, X_copy , y_copy)
    if error : results = pd.Series(results,index=y_copy.index)-y[y.columns[0]]
    all_predictions.append(pd.DataFrame(results,index=y_copy.index,columns=['Run '+str(i)]).sort_index())
  return pd.concat(all_predictions,axis=1)

def produce_predictions(X,y,model, rs=23):  
  #X and y are DataFrames
  X_copy = X.sample(frac=1, replace=True, random_state=rs+i)
  y_copy = y.sample(frac=1, replace=True, random_state=rs+i)
  
  results = pd.Series(cross_val_predict(model, X_copy , y_copy), index= X_copy.index)
  return results
  
def predict_and_save(X,y,model, rs=23, path='./test/test'):
  predictions = produce_predictions(X,y,model, rs=rs)
  predictions.to_csv(path+'.csv')
  return predictions

def plot_yt_ypred_10CV10F(y_true, predictions, dir_path='dummy',file_name='dummy_F_1'):
  if dir_path == 'dummy' : dir_path=os.path.join(os.getcwd(),'dummy_plots/dummy_f')
  if not os.path.isdir(dir_path): os.makedirs(dir_path)
  
  predictions = pd.DataFrame(predictions.loc[y_true.index].mean(axis=1),columns=['Predictions'])
  plt.clf()
  '''
  ax = sns.scatterplot(x=y_true.columns[0], y='Predictions', data=pd.concat([y_true,predictions],axis=1))
  ax.scatter(y_true, predictions, edgecolors=(0, 0, 0))
  '''
  
  fig, ax = plt.subplots()
  g = sns.lmplot(x=y_true.columns[0], y='Predictions', data=pd.concat([y_true,predictions],axis=1))  
  ax = g.axes[0][0]
  ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
  ax.set_xlabel('Measured')
  ax.set_ylabel('Predicted')
  
  plt.savefig(dir_path+'/'+file_name+'target_pred.png')

def plot_yerr_features_10CV10_F(error, feature, dir_path='dummy',file_name='dummy_F_1'):
  if dir_path == 'dummy' : dir_path=os.path.join(os.getcwd(),'dummy_plots/dummy_f')
  if not os.path.isdir(dir_path): os.makedirs(dir_path)
  error = pd.DataFrame(error.loc[feature.index].mean(axis=1),columns=['Error'])
  plt.clf()
  ax = sns.scatterplot(x=feature.columns[0], y='Error', data=pd.concat([feature,error],axis=1))
  plt.tight_layout()
  plt.xticks(rotation=45)
  '''
  fig, ax = plt.subplots()
  ax.scatter(y_true, predictions, edgecolors=(0, 0, 0))
  ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
  ax.set_xlabel('Measured')
  ax.set_ylabel('Predicted')
  '''
  plt.savefig(dir_path+'/'+file_name+'.png')
  
def plot_yerr_features_10CV10_F_log(error, feature, dir_path='dummy',file_name='dummy_F_1'):
  if dir_path == 'dummy' : dir_path=os.path.join(os.getcwd(),'dummy_plots/dummy_f')
  if not os.path.isdir(dir_path): os.makedirs(dir_path)
  error = pd.DataFrame(error.loc[feature.index].mean(axis=1),columns=['Error'])
  plt.clf()
  #ax = sns.scatterplot(x=feature.columns[0], y='Error', data=pd.concat([feature,error],axis=1))
  ax = sns.regplot(x=feature.columns[0], y='Error', data=pd.concat([feature,error],axis=1), logistic=True)
  plt.xticks(rotation=45)
  plt.tight_layout()
  '''
  fig, ax = plt.subplots()
  ax.scatter(y_true, predictions, edgecolors=(0, 0, 0))
  ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
  ax.set_xlabel('Measured')
  ax.set_ylabel('Predicted')
  '''
  plt.savefig(dir_path+'/'+file_name+'.png')


