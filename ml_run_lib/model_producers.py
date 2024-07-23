# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:48:54 2019

@author: Mathieu Ehlinger
Part of the MetaExperiment Project

"""

import pandas as pd
from numpy.random import seed


#from multiprocessing import Pool as ThreadPool
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
#multiprocessing.set_start_method('forkserver')

import subprocess

from ml_run_lib.experiments_functions_new import make_a_ds_run_name_fp

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, make_scorer
import itertools
import tpot
import pickle
import copy
import os
from shutil import copyfile

from tpot import TPOTClassifier
from tpot import TPOTRegressor

#results = pool.map(my_function, my_array)

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
        t_pot = TPOTClassifier(generations=gen, population_size=pop,
                                   verbosity=verbosity,n_jobs=proc, scoring=scorer,
                                   config_dict=config_dict, random_state=rand,
                                   periodic_checkpoint_folder=(name+'_model_temp'),
#                                   early_stop=7
                                   )
    else :
        print('Regressor search running ...')
        t_pot = TPOTRegressor(generations=gen, population_size=pop,
                                   verbosity=verbosity,
                                    n_jobs=proc,
                                   scoring=scorer,
                                   random_state=rand,
                                   config_dict=config_dict,
                                   periodic_checkpoint_folder=(name+'_model_temp'),
#                                   early_stop=7
                                   )
                                   
    t_pot.fit(features, targets)
    #t_pot.score(X_test, y_test)
    
    x=t_pot.fitted_pipeline_
    
    if save :
        t_pot.export((name+'_final_model.py'))

        with open((name+'_train_var.pickle'), 'wb') as pickle_file:
            pickle.dump({'features': features,'targets':targets,
                         'gen' : gen, 'pop': pop, 'name':name,
                         'function' : scorer,
                         'model':x},
                         pickle_file)
        try:
          with open((name+'_complete_tpot.pickle'), 'wb') as pickle_file:
              pickle.dump({'complete_tpot_obj':t_pot},
                           pickle_file)
        except:
          print('Pickling complete TPOT object failed - Skipping this step')            
    return t_pot,x

def fp_for_mt_tpot(df_o,target_ds, exp_path='experiment', target='ds',rand=23, gen=20, classification=True,scorer=None, tpot_proc=1, pop =300,naming_func=make_a_ds_run_name_fp ,config_dict=None):
    #first phase experiment function, designed for multithreading
    #df is from the DataSet class here
    df = df_o.df_data
    seed(rand)
    result=dict()
    print('Experience name :',df_o.ds_name,'\nPopulation : ',pop,' - Generations : ',gen,' - Processors : ',tpot_proc)
    #X_train, X_test, y_train, y_test = train_test_split(df, target_ds, test_size=0.33, random_state=rand)
    #print('Naming function :',naming_func)
    name = naming_func(exp_path, df_o.ds_name)

    #model, fitted=train_tpot_ft(features=df,targets=target_ds, gen=gen, pop=pop,
    train_tpot_ft(features=df,targets=target_ds, gen=gen, pop=pop,
        name=name, scorer=scorer, save = True,
        proc=tpot_proc,target=target, classification=classification,
        config_dict=config_dict, rand=rand)
    #result = fitted.score(X_test, y_test)
    #pd.DataFrame.from_dict(result).to_csv(str(name+'_tpot_runs_results.csv'))
    #return [df_o.ds_name, model]
    
def multithreading_tpot(ds_library, target_ds, exp_path='experiment', target='ds', classification=True,scorer=None,rand=23,arg_dict=dict(),naming_func=make_a_ds_run_name_fp):
    #Reception of the arg_dict is done here
    if 'threadp' in arg_dict.keys() : threadp = arg_dict['threadp']
    else : threadp = 4
    if 'gen' in arg_dict.keys() : gen = arg_dict['gen']
    else : gen = 20
    if 'pop' in arg_dict.keys() : pop = arg_dict['pop']
    else : pop = 300
    if 'tpot_proc' in arg_dict.keys() : tpot_proc = arg_dict['tpot_proc']
    else : tpot_proc = 20
    if 'config_dict' in arg_dict.keys() : config_dict = arg_dict['config_dict']
    else : config_dict = None
    #Some chatting
    print('Multithreaded TPOT experiment initiated ! Running on ', str(len(ds_library.keys())),' libraries with ',threadp,' threads and ', tpot_proc,' processors.') 
    pool = ThreadPool(threadp)
    #Lets go !
    print('Naming function :',naming_func)
    '''
    models = pool.starmap(fp_for_mt_tpot, zip(list(ds_library.values()), itertools.repeat(target_ds),itertools.repeat(exp_path),\
      itertools.repeat(target),itertools.repeat(rand), itertools.repeat(gen),\
      itertools.repeat(classification), itertools.repeat(scorer), itertools.repeat(tpot_proc), itertools.repeat(pop), 
      itertools.repeat(naming_func),itertools.repeat(config_dict) ))
    '''
    pool.starmap(fp_for_mt_tpot, zip(list(ds_library.values()), itertools.repeat(target_ds),itertools.repeat(exp_path),\
      itertools.repeat(target),itertools.repeat(rand), itertools.repeat(gen),\
      itertools.repeat(classification), itertools.repeat(scorer), itertools.repeat(tpot_proc), itertools.repeat(pop), 
      itertools.repeat(naming_func),itertools.repeat(config_dict) ))
    end_dict=dict()
    for key, model in models :
      end_dict[key]=model
      
    return end_dict
    
def tpot_one_thread(ds_library, target_ds, exp_path='experiment', target='ds', classification=True,scorer=None,rand=23,arg_dict=dict(),naming_func=make_a_ds_run_name_fp):
    #Reception of the arg_dict is done here
    if 'threadp' in arg_dict.keys() : threadp = arg_dict['threadp']
    else : threadp = 4
    if 'gen' in arg_dict.keys() : gen = arg_dict['gen']
    else : gen = 20
    if 'pop' in arg_dict.keys() : pop = arg_dict['pop']
    else : pop = 300
    if 'tpot_proc' in arg_dict.keys() : tpot_proc = arg_dict['tpot_proc']
    else : tpot_proc = 20
    if 'config_dict' in arg_dict.keys() : config_dict = arg_dict['config_dict']
    else : config_dict = None
    #Some chatting
    print('Single-thread TPOT experiment initiated ! Running on ', str(len(ds_library.keys())),' libraries with a single threads and ', tpot_proc,' processors.') 
    #pool = ThreadPool(threadp)
    #Lets go !
    '''
    df_o,target_ds, exp_path='experiment', target='ds',rand=23, gen=20, classification=True,scorer=None, tpot_proc=1, pop =300,naming_func=make_a_ds_run_name_fp ,config_dict=None
    '''
    print('Naming function :',naming_func)
    for dataset_name in ds_library :
      fp_for_mt_tpot(df_o=ds_library[dataset_name],target_ds=target_ds,exp_path=exp_path,\
      target=target,rand=rand, gen=gen, classification=classification, scorer=copy.deepcopy(scorer), tpot_proc=tpot_proc, pop=pop,\
      naming_func=naming_func,config_dict=config_dict)
    
    end_dict=dict()
    for key, model in models :
      end_dict[key]=model
      
    return end_dict

def multithreading_tpot_cv(df_o, exp_name, exp_norm_file='experiment', target='ds',rand=23, n_splits=3, classification=True):
    #df is from the DataSet class here
    df = df_o.df_data

    result=dict()
    kf = KFold(n_splits=n_splits)
    i=0

    for train_index, test_index in kf.split(df.drop(target,axis=1).values):

        X_train, X_test = df.drop(target,axis=1).values[train_index], \
            df.drop(target,axis=1).values[test_index]

        y_train, y_test = df[target].values[train_index], \
            df[target].values[test_index]
            
        name = make_a_ds_run_name_fp(exp_name, df_o.name+str(i))

        model=train_tpot_ft(features=X_train,targets=y_train, gen=5, pop=300,
            name=name, scorer=make_scorer(balanced_accuracy_score), save = True,
            proc=2,target=target, classification=classification,
            config_dict=None, rand=rand)
        result[str(i)] = x.score(X_test, y_test)
        i+=1

    pd.Dataframe(results).to_csv(str(name+'_tpot_runs_results.csv'))

def blank_run_writer(features,targets, gen, pop, name, scorer, save = True,
               proc=1,target='ds', classification=True,
               config_dict=None, rand=23):
    features.to_csv(path_or_buf=(name + '_features.csv'), index = True)
    targets.to_csv(path_or_buf=(name + '_targets.csv'), index = True)
    pickle.dump( {'gen':gen, 'pop':pop, 'name':name, 'scorer':scorer, 'save':save,
               'proc':proc,'target' :target, 'classification':classification,
               'config_dict':config_dict, 'rand':rand}, open( name + '_configs.pickle', "wb" ) )

def blank_run_writer_tpot(features,targets, gen, pop, name, scorer, save = True,
               proc=1,target='ds', classification=True,
               config_dict=None, rand=23):
    features.to_csv(path_or_buf=(name + '_features.csv'), index = True)
    targets.to_csv(path_or_buf=(name + '_targets.csv'), index = True)
    pickle.dump( {'gen':gen, 'pop':pop, 'name':name, 'scorer':scorer, 'save':save,
               'proc':proc,'target' :target, 'classification':classification,
               'config_dict':config_dict, 'rand':rand}, open( name + '_configs.pickle', "wb" ) )
    
    src = str(os.path.dirname(os.path.abspath(__file__))) + '/model_search_templates/' + 'tpot_template.py'
    
    dst = name + '_tpot_run.py'
    
    print('Source :',src,'\nDestination :',dst)
    
    copyfile(src, dst)


def fp_for_mt_blank_run(df_o,target_ds, exp_path='experiment', target='ds',rand=23, gen=20, classification=True,scorer=None, tpot_proc=1, pop =300,naming_func=make_a_ds_run_name_fp ,config_dict=None):
    #first phase experiment function, designed for multithreading
    #df is from the DataSet class here
    df = df_o.df_data
    seed(rand)
    result=dict()
    print('Experience name :',df_o.ds_name,'\nPopulation : ',pop,' - Generations : ',gen,' - Processors : ',tpot_proc)
    #X_train, X_test, y_train, y_test = train_test_split(df, target_ds, test_size=0.33, random_state=rand)
    #print('Naming function :',naming_func)
    name = naming_func(exp_path, df_o.ds_name)

    #model, fitted=train_tpot_ft(features=df,targets=target_ds, gen=gen, pop=pop,
    blank_run_writer(features=df,targets=target_ds, gen=gen, pop=pop,
        name=name, scorer=scorer, save = True,
        proc=tpot_proc,target=target, classification=classification,
        config_dict=config_dict, rand=rand)
    
def fp_for_mt_tpot_blank_run(df_o,target_ds, exp_path='experiment', target='ds',rand=23, gen=20, classification=True,scorer=None, tpot_proc=1, pop =300,naming_func=make_a_ds_run_name_fp ,config_dict=None):
    #first phase experiment function, designed for multithreading
    #df is from the DataSet class here
    df = df_o.df_data
    seed(rand)
    result=dict()
    print('Experience name :',df_o.ds_name,'\nPopulation : ',pop,' - Generations : ',gen,' - Processors : ',tpot_proc)
    #X_train, X_test, y_train, y_test = train_test_split(df, target_ds, test_size=0.33, random_state=rand)
    #print('Naming function :',naming_func)
    name = naming_func(exp_path, df_o.ds_name)

    #model, fitted=train_tpot_ft(features=df,targets=target_ds, gen=gen, pop=pop,
    blank_run_writer_tpot(features=df,targets=target_ds, gen=gen, pop=pop,
        name=name, scorer=scorer, save = True,
        proc=tpot_proc,target=target, classification=classification,
        config_dict=config_dict, rand=rand)
    

def multithreading_blank_run(ds_library, target_ds, exp_path='experiment', target='ds', classification=True,scorer=None,rand=23,arg_dict=dict(),naming_func=make_a_ds_run_name_fp):
    #Reception of the arg_dict is done here
    if 'threadp' in arg_dict.keys() : threadp = arg_dict['threadp']
    else : threadp = 4
    if 'gen' in arg_dict.keys() : gen = arg_dict['gen']
    else : gen = 20
    if 'pop' in arg_dict.keys() : pop = arg_dict['pop']
    else : pop = 300
    if 'tpot_proc' in arg_dict.keys() : tpot_proc = arg_dict['tpot_proc']
    else : tpot_proc = 20
    if 'config_dict' in arg_dict.keys() : config_dict = arg_dict['config_dict']
    else : config_dict = None
    #Some chatting
    print('Multithreaded blank TPOT run initiated ! Running on ', str(len(ds_library.keys())),' libraries with ',threadp,' threads and ', tpot_proc,' processors.') 
    pool = ThreadPool(threadp)
    #Lets go !
    print('Naming function :',naming_func)
    '''
    models = pool.starmap(fp_for_mt_tpot, zip(list(ds_library.values()), itertools.repeat(target_ds),itertools.repeat(exp_path),\
      itertools.repeat(target),itertools.repeat(rand), itertools.repeat(gen),\
      itertools.repeat(classification), itertools.repeat(scorer), itertools.repeat(tpot_proc), itertools.repeat(pop), 
      itertools.repeat(naming_func),itertools.repeat(config_dict) ))
    '''
    pool.starmap(fp_for_mt_blank_run, zip(list(ds_library.values()), itertools.repeat(target_ds),itertools.repeat(exp_path),\
      itertools.repeat(target),itertools.repeat(rand), itertools.repeat(gen),\
      itertools.repeat(classification), itertools.repeat(scorer), itertools.repeat(tpot_proc), itertools.repeat(pop), 
      itertools.repeat(naming_func),itertools.repeat(config_dict) ))
    end_dict=dict()
      
    return end_dict

def multithreading_tpot_blank_run(ds_library, target_ds, exp_path='experiment', target='ds', classification=True,scorer=None,rand=23,arg_dict=dict(),naming_func=make_a_ds_run_name_fp):
    #Reception of the arg_dict is done here
    if 'threadp' in arg_dict.keys() : threadp = arg_dict['threadp']
    else : threadp = 4
    if 'gen' in arg_dict.keys() : gen = arg_dict['gen']
    else : gen = 20
    if 'pop' in arg_dict.keys() : pop = arg_dict['pop']
    else : pop = 300
    if 'tpot_proc' in arg_dict.keys() : tpot_proc = arg_dict['tpot_proc']
    else : tpot_proc = 20
    if 'config_dict' in arg_dict.keys() : config_dict = arg_dict['config_dict']
    else : config_dict = None
    #Some chatting
    print('Multithreaded blank TPOT run initiated ! Running on ', str(len(ds_library.keys())),' libraries with ',threadp,' threads and ', tpot_proc,' processors.') 
    pool = ThreadPool(threadp)
    #Lets go !
    print('Naming function :',naming_func)
    '''
    models = pool.starmap(fp_for_mt_tpot, zip(list(ds_library.values()), itertools.repeat(target_ds),itertools.repeat(exp_path),\
      itertools.repeat(target),itertools.repeat(rand), itertools.repeat(gen),\
      itertools.repeat(classification), itertools.repeat(scorer), itertools.repeat(tpot_proc), itertools.repeat(pop), 
      itertools.repeat(naming_func),itertools.repeat(config_dict) ))
    '''
    pool.starmap(fp_for_mt_tpot_blank_run, zip(list(ds_library.values()), itertools.repeat(target_ds),itertools.repeat(exp_path),\
      itertools.repeat(target),itertools.repeat(rand), itertools.repeat(gen),\
      itertools.repeat(classification), itertools.repeat(scorer), itertools.repeat(tpot_proc), itertools.repeat(pop), 
      itertools.repeat(naming_func),itertools.repeat(config_dict) ))
    end_dict=dict()
      
    return end_dict

def run_as_subprocess(cmd):
    process = subprocess.Popen(cmd.split(' '))
    process.wait()

def tpot_in_subprocesses(ds_library,exp_path, naming_func, nb_parallel_runs=4,*args, **kwargs):
    #multithreading_tpot_blank_run(ds_library, target_ds, exp_path='experiment', target='ds', classification=True,scorer=None,rand=23,arg_dict=dict(),naming_func=make_a_ds_run_name_fp)
    #cmds = ['python '+ exp_path+'/exp_dirs/'+ ds_library[i].ds_name+'/first_phase/'+ds_library[i].ds_name+'_tpot_run.py' for i in ds_library]
    cmds = ['python '+ naming_func(exp_path,ds_library[i].ds_name,raise_if_exists=False)+'_tpot_run.py' for i in ds_library]
    pool = ThreadPool(nb_parallel_runs)
    pool.map(run_as_subprocess, cmds)
    #return cmds