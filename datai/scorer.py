# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:18:57 2018

@author: Mathieu
"""

import pandas as pd
import numpy as np

#from sklearn.metrics import accuracy_score, log_loss,auc, balanced_accuracy_score
from sklearn.metrics import accuracy_score, log_loss,auc
from sklearn.metrics import f1_score, cohen_kappa_score, make_scorer
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

#----------------------------------
def classifier (x):
    if x==0:
        return 'Not depressive'
    else:
        return 'Depressive'
    
#----------------------------------
    
def roc_auc_scorer(Y_test, Y_pred, average = 'macro'):
    Y_test=[1 if x=='Depressive' else x for x in Y_test]
    Y_test=[0 if x=='Not depressive' else x for x in Y_test]
    Y_pred=[1 if x=='Depressive' else x for x in Y_pred]
    Y_pred=[0 if x=='Not depressive' else x for x in Y_pred]
    return roc_auc_score(Y_test, Y_pred,average = average)


def f1_scorer(Y_test, Y_pred, mean_type='macro'):
    #Return f1_score, without balancing between classes
    return f1_score(Y_test,Y_pred,average = mean_type )

def kappa(Y_test, Y_pred, mean_type='macro'):
    #Returns cohen kappa score between classes
    Y_test=[1 if x=='Depressive' else x for x in Y_test]
    Y_test=[0 if x=='Not depressive' else x for x in Y_test]
    Y_pred=[1 if x=='Depressive' else x for x in Y_pred]
    Y_pred=[0 if x=='Not depressive' else x for x in Y_pred]
    return cohen_kappa_score(Y_test,Y_pred)

#----------------------------------
# Score can only be used to quantify Sensitivity and Specificity for DS Scoring

def score(Y_test, Y_pred, verbosity=0):
    #mean_absolute_error(Y_test,Y_pred)
    Y_pred = pd.Series(Y_pred) # pred = pd.Series for character support

    if Y_test.iloc[0]==0 or Y_test.iloc[0]==1:
        Y_test= Y_test.apply(classifier)
        Y_pred= Y_pred.apply(classifier)
    Y_test=Y_test.tolist()
    Y_pred=Y_pred.tolist()
    zipped=zip(Y_test,Y_pred)
    Sensitivity, Specificity = (0,0)
    n_dep,  pp_diag,n_ndep, np_diag = (0,0,0,0)
    
    for x in list (zipped):
        #print(x[0],x[1])
        if x[0] == 'Depressive':
            n_dep+=1
        else :
            n_ndep+=1

        if x[0]==x[1] and x[1]=='Depressive':
            pp_diag +=1
        elif x[0]==x[1] and x[1]=='Not depressive':
            np_diag +=1
    
    if n_dep!=0 :
        Sensitivity = np.float64(pp_diag/n_dep)
    else :
        Sensitivity = np.nan
        
    if n_ndep!= 0 :    
        Specificity = np.float64(np_diag/n_ndep)
    else :
        Specificity = np.nan
        
    if verbosity ==1 :
        print ("Patients dépressifs:", n_dep,"Patients non dépressifs:",n_ndep)
        print("Diagnostics positifs corrects :",pp_diag)
        print("Diagnostics négatifs corrects :",np_diag)
        print("Sensitivity = ",Sensitivity,"Specificity =",Specificity,"\n\n")
        


    return {'sens':Sensitivity, 'spec':Specificity}

# Score output doesn't depend on 'score' - can be used for other scores
# Sens and spec dependent on 'score', should only be used for ds 

def score_model(data,model, target = 'ds',
                score=score, verbosity=1, scorer=f1_scorer, 
                return_algorythms=False, cv=3):
    
    X = data.drop(target,axis=1)
    Y = data[target]
    
    if return_algorythms: produced_algorithms=[]
    
    results = pd.DataFrame(columns=['sens', 'spec', 'score']) 
    
    for i in range(10) :
        
        if return_algorythms: produced_algorithms.append([])
        
        kf = KFold(n_splits=cv,shuffle=True)
        exp = pd.DataFrame(columns=['sens', 'spec', 'score'])
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            m = model
            m.fit(X_train,Y_train)
            Y_pred = m.predict(X_test)
        
            out = score(Y_test, Y_pred,verbosity = verbosity)
        
            c= scorer(Y_test, Y_pred)

            exp = exp.append({'sens': out['sens'], 'spec':out['spec'], 
                'score':c}, ignore_index=True)
                
            if return_algorythms: produced_algorithms[i].append(m)

        results = results.append({'sens': exp['sens'].mean(), 
                                  'spec':exp['spec'].mean(), 
                                  'score':exp['score'].mean()},
                                ignore_index=True)
    print(results)   
    ret=pd.DataFrame({'sens': results['sens'].mean(), 
                                'spec':results['spec'].mean(), 
                                'score':results['score'].mean()},index=[0])
    print(ret)
    
    if not(return_algorythms):
        return ret
    else :
        return ret, produced_algorithms
#----------------------------------

def nichts(df, drop):
    return df.drop(drop, axis =1 ) 

#----------------------------------

def sens_spec_score(Y_test, Y_pred):
    
    #returns mean value of sens / Spec
    #intended for 2 classes
    Y_test=Y_test.tolist()
    Y_pred=Y_pred.tolist()
    zipped=zip(Y_test,Y_pred)
    n_dep,  pp_diag,n_ndep, np_diag = (0,0,0,0)
    
    for x in list (zipped):
        
        if x[0] == 'Depressive':
            n_dep+=1
        else :
            n_ndep+=1

        if x[0]==x[1] and x[1]=='Depressive':
            pp_diag +=1
        elif x[0]==x[1] and x[1]=='Not depressive':
            np_diag +=1
            
    if n_dep!=0 :
        Sensitivity = np.float64(pp_diag/n_dep)
    else :
        Sensitivity = 0
        
    if n_ndep!= 0 :    
        Specificity = np.float64(np_diag/n_ndep)
    else :
        Specificity = 0
        
    return np.float64(((Sensitivity * Specificity)*2)/
                       (Sensitivity + Specificity))

#----------------------------------
scoring={'accuracy':accuracy_score,
                                   'cohen kappa score':kappa,
#                                   'ROC AUC':auc,
                                   'precision':    lambda x,y :precision_score(x,y, average='macro'),
                                   'recall':       lambda x,y :recall_score(x,y, average='macro'),
                                   'f1 score':f1_score,
                                   #'balanced accuracy':balanced_accuracy_score,
                                   'log loss':log_loss}

def scoring_it(Y_true, Y_pred, scorers):
    results = dict()
    for scorer in scorers.keys():
        results[scorer]=scoring[scorer](Y_true, Y_pred)
    return results