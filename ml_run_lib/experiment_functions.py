# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:56:15 2018

@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""

import os
import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer

from ml_run_lib.tpot_run import train_tpot

from datai.scorer import f1_scorer
from datai.scorer import score_model


#----------------------------------

#Creates the evaluation function, shapes the datas and creates + trains the model


def pipeline_maker(df, name='test', scorer=f1_scorer, classification=True,
                   pop = 500, gen = 10, th = 19,save=True, target='ds',
                   greater_is_better=True, config_dict=None, rand = 23):
    
    score_pipe=make_scorer(scorer, greater_is_better=greater_is_better)
    
    model = train_tpot(df, gen, pop, name, score_pipe, save, target=target,
                       classification=classification,config_dict=config_dict, rand=rand)
    
    return model

        
#------------------------------



def pipeline_eval (name='test', scorer=f1_scorer,
                   model=None, df= None,
                   save=True, verbosity=1, target='ds', 
                   classification = True, return_algorythms=False, cv=3, rand = 23):
    if classification :
#        if save :
#            with open((name+'.cfg'), 'rb') as pickle_file:
#                dic=pickle.load(pickle_file)    
#                if verbosity >0:
#                    print("Real test : ")
#                Pred = model.predict(dic['X_test'])
#                r1 = score(dic['y_test'],Pred, verbosity =verbosity )
#                r1['score']=scorer(dic['y_test'],Pred)
#        else :
#            r1 = None
        np.random.seed(rand)
        if verbosity >0:
            print(" Reroll : ")
        r2 = score_model(df, model, cv=cv, target=target, verbosity=verbosity, scorer=scorer,return_algorythms=return_algorythms)
    
        if not(return_algorythms): return {'r2':r2}
        else : return {'r2':r2[0]},r2[1] 


#------------------------------

def assign_filename (name, save):
    file_name=name
    if save :
        file_name = './'+ name+'/'   
        if not os.path.exists(name):
            os.makedirs(name)
            
        else :
            file_id = str(datetime.datetime.now())
            file_id = file_id.replace('.','_')
            file_id = file_id.replace(' ','-')
            file_id = file_id.replace(':','_')
            file_id = file_id[:-7]

            os.makedirs((name+' '+file_id))
            file_name = './'+ name+' '+file_id+'/'
    return file_name
    

def make_an_excel (name, dir_path):

    writer = pd.ExcelWriter((dir_path+name+'.xlsx'))
    
    for filename in os.listdir(dir_path):
#        print(filename)
        if filename.endswith(".csv") : 
#            print('check 2')
#            print(filename)
            t = pd.read_csv((dir_path+filename))
            t.to_excel(writer, sheet_name = filename[:-4], index=False)
    writer.save()
#------------------------------
