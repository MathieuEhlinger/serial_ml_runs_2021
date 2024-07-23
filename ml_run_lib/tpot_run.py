# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:16:35 2018

@author: Mathieu
"""

import tpot
import pickle

from tpot import TPOTClassifier
from tpot import TPOTRegressor

#----------------------------------

#TPOT pipeline generating & training function
# The df & training + scoring parameter are saved in a pickle dictionnary
    
def train_tpot(df, gen, pop, name, scorer, save = True,
               proc=1,target='ds', classification=True,
               config_dict=None, rand=23):
        
    features = df.drop(target,axis=1)
    target = df[target]
    
    print('---------------------')
    
    if save :    
        features.to_csv(path_or_buf=(name + ' features.csv'), index = False)
        target.to_csv(path=(name + ' targets.csv'), index = False)

#    features = features.values  #Making sure the index names have no influence
#    target = target.values

#    X_train, X_test, y_train, y_test = train_test_split(
#            features,target,
#            train_size=0.75, test_size=0.25)
        
    if classification :    
        print('Classifier search running ...')
        t_pot = TPOTClassifier(generations=gen, population_size=pop, 
                                   verbosity=2,n_jobs=proc, scoring=scorer, 
                                   config_dict=config_dict, random_state=rand,
                                   periodic_checkpoint_folder=(name+'model_temp'),
#                                   early_stop=7
                                   )
    else :
        print('Regressor search running ...')
        t_pot = TPOTRegressor(generations=gen, population_size=pop, 
                                   verbosity=2,
                                    n_jobs=proc, 
#                                    scoring=scorer,
                                   random_state=rand,
#                                   config_dict=config_dict, 
                                   periodic_checkpoint_folder=(name+'model_temp'),
#                                   early_stop=7
                                   )

    t_pot.fit(features, target)
    #t_pot.score(X_test, y_test)
    
    x=t_pot.fitted_pipeline_
    if save :
        
                
        t_pot.export((name+'model.py'))
        
        with open((name+' train_var.pickle'), 'wb') as pickle_file:
            pickle.dump({'df': df, 'gen' : gen, 'pop': pop, 'name':name, 
                         'function' : scorer,'X_test':features,
                         'y_test':target, 'model':x }, 
                         pickle_file)

    return x

#----------------------------------