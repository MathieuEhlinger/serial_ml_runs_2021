import sys

import pathlib
import itertools
import tpot
import sklearn
import pickle
import copy
import os
import pandas as pd

from tpot import TPOTClassifier
from tpot import TPOTRegressor

def train_tpot_ft(features,targets, gen, pop, name, scorer, save = True,
               proc=1,target='ds', classification=True,
               config_dict=None, rand=23,verbosity=1,save_entry=True):
    if save_entry : features.to_csv(path_or_buf=(name + '_features.csv'), index = True)
    if save_entry : targets.to_csv(path_or_buf=(name + '_targets.csv'), index = True)

    print('---------------------')

    #if save :
    #    pd.DataFrame(features).to_csv(path_or_buf=(name + ' features.csv'), index = True)
    #    pd.DataFrame(targets).to_csv(path_or_buf=(name + ' targets.csv'), index = True)



#    features = features.values  #Making sure the index names have no influence
#    target = target.values

#    X_train, X_test, y_train, y_test = train_test_split(
#            features,target,
#            train_size=0.75, test_size=0.25)

    if classification :
        print('Classifier search running ...')
        print('Proc. :', proc,' - Verbosity :',verbosity)
        t_pot = TPOTClassifier(generations=gen, population_size=pop,
                                   verbosity=verbosity,n_jobs=proc, scoring=scorer,
                                   config_dict=config_dict, random_state=rand,
                                   periodic_checkpoint_folder=(name+'_model_temp'),
#                                   early_stop=7
                                   )
    else:
        print('Regressor search running ...')
        print('Proc. :', proc,' - Verbosity :',verbosity)
        t_pot = TPOTRegressor(generations=gen, population_size=pop,
                                   verbosity=verbosity,
                                   n_jobs=proc,
                                   scoring=scorer,
                                   random_state=rand,
                                   config_dict=config_dict,
                                   periodic_checkpoint_folder=(name+'_model_temp'),
#                                   early_stop=7
                                   )

    t_pot.fit(features, targets[target])
    #t_pot.score(X_test, y_test)

    x=t_pot.fitted_pipeline_
    if save :


        t_pot.export((name+'_final_model.py'))
        
        with open((name+'_main_lib_version.txt'), 'w') as infile :
            infile.write('TPOT version : ' + str(tpot.__version__) + '\n' + 'scikit-klearn version : ' + str(sklearn.__version__))
          
        with open((name+'_train_var.pickle'), 'wb') as pickle_file:
            pickle.dump({'features': features,'targets':targets,
                         'gen' : gen, 'pop': pop, 'name':name,
                         'function' : scorer,'X_test':features,
                         'y_test':target, 'model':x},
                         pickle_file)
        with open((name+'_tpot_ei.pickle'), 'wb') as pickle_file:
            pickle.dump({'Evaluated_individuals':tpot.evaluated_individuals_},
                         pickle_file)
        with open((name+'_tpot_save.pickle'), 'wb') as pickle_file:
            pickle.dump({'complete_tpot_obj':t_pot},
                         pickle_file)

    return t_pot,x

if __name__ == '__main__' :
    exp_path  = str(os.path.dirname(os.path.abspath(__file__)))
    exp_name  = str(exp_path.split('/')[-2])
    name = exp_path+'/'+exp_name
    features  = pd.DataFrame.from_csv(name+'_features.csv')
    targets   = pd.DataFrame.from_csv(name+'_targets.csv')
    
    with open(name + '_configs.pickle', "rb" ) as pickle_file:
        pic = pickle.load( pickle_file )
        '''
        {'gen':gen, 'pop':pop, 'name':name, 'scorer':scorer, 'save':save,
                   'proc':proc,'target' :target, 'classification':classification,
                   'config_dict':config_dict, 'rand':rand}, open( name + '_configs.pickle', "wb" )
        '''
        scorer    = pic['scorer']
        gen = pic['gen']
        classification = pic['classification']
        tpot_proc = pic['proc']
        pop = pic['pop']
        rand = pic['rand']
        config_dict= pic['config_dict']
        target= pic['target']
    
    train_tpot_ft(features=features,targets=targets, gen=gen,pop=pop,name=name,scorer=scorer,\
          classification=classification, proc=tpot_proc, save=True,save_entry=False ,config_dict=config_dict,rand=rand,target=target,verbosity=3)