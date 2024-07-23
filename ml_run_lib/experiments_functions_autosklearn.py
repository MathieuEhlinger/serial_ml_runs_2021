# -*- coding: utf-8 -*-
"""
@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""
from autosklearn.metrics import balanced_accuracy
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *

import pandas as pd

def last_cv_test_askl(ensemble, df_hm2, target='ds', rand=23, folds = 10, cv= 10):
    df, ensemble_copy, results, results_all = df_hm2.copy(), copy.deepcopy(ensemble), dict(), list()
    df_res=list()
    for i in range(cv):
        try : 
            kf = KFold(n_splits=folds, shuffle=True, random_state=int(rand+i))
            for train_index, test_index in kf.split(df.drop(target,axis=1).values):
            
                X_train, X_test = df.drop(target,axis=1).values[train_index], \
                    df.drop(target,axis=1).values[test_index]
            
                y_train, y_test = df[target].values[train_index], \
                    df[target].values[test_index]
                
                ensemble_copy.refit(X_train, y_train)
                pred = ensemble_copy.predict(X_test)
                results['accuracy'] = accuracy_score(pred,y_test)
                results['precision'] = precision_score(pred,y_test)
                results['recall_score'] = recall_score(pred,y_test)
                results_all.append(copy.deepcopy(results))
            df_res.append(pd.DataFrame.from_dict(results_all).mean().to_dict())
        except :
            print ('Crash at iteration:',i)
    return df_res

def last_train_test_askl(ensemble, df_hm1,df_hm2, target='ds'):
    ensemble_copy= copy.deepcopy(ensemble)
    ensemble_copy.refit(df_hm1.drop(target,axis=1), df_hm1[target])
    pred = ensemble_copy.predict(df_hm2.drop(target,axis=1))
    
    print(accuracy_score(pred, df_hm2[target]))
    print(precision_score(pred, df_hm2[target]))
    print(recall_score(pred, df_hm2[target]))
    
    return (accuracy_score(pred, df_hm2[target]),precision_score(pred, df_hm2[target]), recall_score(pred, df_hm2[target]))


def fast_auto_skl_exp(df, name, target='ds',rand=23,time=600,per_run_time=60):
  global gaskl_model_catcher 
  X_train, X_test, y_train, y_test = \
  sklearn.model_selection.train_test_split(df.drop(target,axis=1), df[target], random_state=rand)
  
  automl = AutoSklearnClassifier(
        time_left_for_this_task=time,
        ml_memory_limit=10240,
        per_run_time_limit=per_run_time,
        tmp_folder=str(name+'/tmp/autosklearn_sp_tmp'),
        output_folder=str(name+'/autosklearn_sp_out'),
        # Do not construct ensembles in parallel to avoid using more than one
        # core at a time. The ensemble will be constructed after auto-sklearn
        # finished fitting all machine learning models.
        ensemble_size=1,
        delete_tmp_folder_after_terminate=False,
        resampling_strategy='cv',
        )
  
  automl.fit(X_train, y_train, dataset_name='n_anat_d')
  pd.DataFrame(automl.cv_results_).to_csv(str(name+'_askl_results.csv'), index=False)
  automl.sprint_statistics()  
  automl.fit_ensemble(y_train)
  automl.refit(X_train, y_train)
  predictions = automl.predict(X_test)
  
  print(automl.sprint_statistics())
  print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
  print("Precision score", sklearn.metrics.precision_score(y_test, predictions))
  print("Recall score", sklearn.metrics.recall_score(y_test, predictions))
  
  pd.DataFrame(automl.cv_results_).to_csv(str(name+'_askl_ensembled_results.csv'), index=False)
  automl.sprint_statistics()
  automl.show_models()
  with open( name+' model.p', "wb" )as file: 
        pickle.dump( automl, file )
  gaskl_model_catcher=copy.deepcopy(automl)
  return automl

def auto_skl_exp(df, name, target='ds',rand=23,time=60):
  global gaskl_model_catcher 
  X_train, X_test, y_train, y_test = \
  sklearn.model_selection.train_test_split(df.drop(target,axis=1), df[target], random_state=rand)
  
  automl = AutoSklearnClassifier(
        exclude_preprocessors=['select_rates'],
        time_left_for_this_task=time,
        ml_memory_limit=10240,
        per_run_time_limit=300,
        tmp_folder=str(name+'/tmp/autosklearn_sp_tmp'),
        output_folder=str(name+'/autosklearn_sp_out'),
        # Do not construct ensembles in parallel to avoid using more than one
        # core at a time. The ensemble will be constructed after auto-sklearn
        # finished fitting all machine learning models.
        ensemble_size=50,
        delete_tmp_folder_after_terminate=False,
    )
  
  automl.fit(X_train, y_train, dataset_name='n_anat_d')
  
  pd.DataFrame(automl.cv_results_).to_csv(str(name+'_askl_results.csv'), index=False)
  automl.sprint_statistics()

  
  automl.fit_ensemble(y_train, ensemble_size=50)
  
  print('Auto sklearn - ensembling enabled')
  print(automl.show_models())
  predictions = automl.predict(X_test)
  print(automl.sprint_statistics())
#  print("Balanced accuracy score", sklearn.metrics.balanced_accuracy_score(y_test, predictions))
  print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
  print("Precision score", sklearn.metrics.precision_score(y_test, predictions))
  print("Recall score", sklearn.metrics.recall_score(y_test, predictions))
  
  pd.DataFrame(automl.cv_results_).to_csv(str(name+'_askl_ensembled_results.csv'), index=False)
  automl.sprint_statistics()
  automl.show_models()
  with open( name+' model.p', "wb" )as file :
    pickle.dump( automl, file )
  gaskl_model_catcher=copy.deepcopy(automl)
  return automl