# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 01:51:08 2018

@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""
import sys, os
sys.path.append('./../datai')

get_cwd = os.getcwd()
os.chdir(os.path.dirname(__file__))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import ml_run_lib.modelslists as modelslists
import pygam
from pygam import LogisticGAM, s, f
import numpy as np

from ml_run_lib.massive_grid_search import grid_searching_through_the_night
from ml_run_lib.tpot_run import train_tpot
from ml_run_lib.check_all_c import check_all_classifiers

from datai.scorer import f1_scorer
from datai.scorer import kappa
from datai.scorer import roc_auc_scorer
from ipywidgets import IntProgress

from sklearn import preprocessing

from sklearn.model_selection import KFold
#from sklearn.metrics import balanced_accuracy_score,make_scorer
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle
#from sklearn.metrics import accuracy_score, log_loss, balanced_accuracy_score,auc
from sklearn.metrics import accuracy_score, log_loss, auc
from sklearn.metrics import f1_score, cohen_kappa_score, make_scorer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

rand=23
standard_classifiers = modelslists.classifier_dict
standard_preprocessors = modelslists.preprocessor_dict
test_classifiers= modelslists.test

iris = datasets.load_iris()
bc = datasets.load_breast_cancer()

df = pd.DataFrame(iris.data)
df['target'] = iris.target
df = shuffle(df, random_state =rand)

df2 = pd.DataFrame(bc.data)
df2['target'] = bc.target
df2 = shuffle(df2, random_state =rand)


def name_target(df):
    df['target']=df.target.apply(lambda x: ('Type '+str(x)))
    return df
    

def grid_searching_it(df, classifier_list=standard_classifiers,
                      directory='grid_searching',name='test',target='target'):
#    df=name_target(df)
    if not os.path.exists(directory):
        os.makedirs(directory)
    grid_searching_through_the_night(df, name=(directory+'/'+name), classifier_list=classifier_list,
                                 target=target, 
                                 scoring={
                                 #'balanced_accuracy_score':balanced_accuracy_score
                                 'f1_score':f1_score},
                                 dict_filter='test', fil=False, rand=23,cv=3)
    
def tpot_test(df, directory ='test', name='name',target='target', config_dict=None):
#    df_c=name_target(df.copy())
    df_c = df.copy()
    if not os.path.exists(directory):
        os.makedirs(directory)
    train_tpot(df_c, 25, 200, (directory+'/'+name) , 
#           make_scorer(balanced_accuracy_score), save = True,
            make_scorer(f1_score), save = True,
            proc=1,target=target, classification=True,
            config_dict=config_dict, rand=23)

def check_cl(df,directory='check_classifiers',name='cc',target='target'):
#    df=name_target(df)
    if not os.path.exists(directory):
            os.makedirs(directory)
    check_all_classifiers(df, file_name=str(directory+'/'+name),rand=23, 
                              scoring={'accuracy':accuracy_score,
                                       'cohen kappa score':kappa,
    #                                   'f1 score':f1_score,
    #                                   'balanced accuracy':balanced_accuracy_score,
    #                                   'log loss':log_loss
                                       } , 
                                        cv=5, save=True, target=target)
    
    
#results3, model3, model32 = pygam_classification(df3c, target='diagnosis')
#results4, model4, model42 = pygam_classification(df4, target='diagnosis')

def pygam_classification(df, target='target',cv= 5,
#                         
                          scorer = f1_score,
                         ):
    features, targets = df.drop(target, axis =1).values, df[target].values
    for i,typ in enumerate(features):
        print(i,' : ',type(typ), ', ', typ.shape)
    for i,typ in enumerate(targets):
        print(i,' : ',type(typ), ', ', typ.shape)
    kf= KFold(n_splits=cv, shuffle=False, random_state=rand)
    for train_index, test_index in kf.split(features):
        model1 = pygam.pygam.LogisticGAM()
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        model1.fit(X_train,y_train)
        predict =model1.predict(X_test)
        print(scorer(y_test, predict))
    print('----------')
    model2=pygam.pygam.LogisticGAM()
    model2 = model2.gridsearch(X_train,y_train)
    predict =model2.predict(X_test)
    print(model2)
    print(scorer(y_test, predict))
#    
#    model = pygam.pygam.LogisticGAM()
    print('----------')
    model=LogisticGAM(callbacks=['deviance', 'diffs', 'accuracy'], 
                                             fit_intercept=True, max_iter=100, 
#        terms=s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12) + s(13) + s(14) + s(15) + s(16) + s(17) + s(18) + s(19) + s(20) + s(21) + s(22) + s(23) + s(24) + s(25) + s(26) + s(27) + s(28) + s(29),
        tol=0.0001, verbose=False)
    print('----------')
    results = cross_validate(model,  df.drop(target, axis =1),\
                                    df[target], cv=cv, scoring=make_scorer(scorer))
    return results, model, model2
#    model.fit(df.drop(target,axis=1),df[target])
#watch_pygam_plots(model32, df3c, target='diagnosis')
#watch_pygam_plots(model42, df4, directory= 'test_pdplot_false_d',target='diagnosis')
def watch_pygam_plots(model,df, directory='test_pdplot', target='target'):
    features=list(df.columns)
    features.remove(target)
    
    if not os.path.exists(directory):
            os.makedirs(directory)
    for i, term in enumerate(model.terms):
        if term.isintercept:
            continue
        XX = model.generate_X_grid(term=i)
        pdep, confi = model.partial_dependence(term=i, X=XX, width=0.95)
        fig = plt.figure()
        plt.plot(XX[:, term.feature], pdep)
        plt.plot(XX[:, term.feature], confi, c='r', ls='--')
        plt.title(features[i])
        fig.savefig(directory+'/'+str(i)+'. '+ features[i]+'.png', dpi=fig.dpi)
        plt.show()
        
os.chdir(get_cwd)