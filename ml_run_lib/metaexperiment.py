# -*- coding: utf-8 -*-
"""
@author: Mathieu Ehlinger
Part of the automating Project
"""
import copy
import os
import pandas as pd
from pandas.plotting import table
import math
import pickle
from os import listdir
from os.path import isfile, join
from datetime import datetime
import numpy as np
from collections import OrderedDict
import shutil
import seaborn as sns
from copy import deepcopy

from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
#from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.utils import shuffle
from sklearn.base import clone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datai.ds_building_functions import DSLibrary, DataSet
import ml_run_lib.model_producers
from ml_run_lib.experiments_functions_new import make_a_meta_exp_name, make_a_ds_run_name_fp, make_a_ds_run_name_2pb, produce_the_namer
#from ml_run_lib.tpot_code_reader import tpot_code_execution, tpot_model_score

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from organiser.repartition import split_in_matching_datasets, db1

class MissingModelError(Exception):
    pass

class UndefinedError(Exception):
    pass

class DictsInSpareDictError(Exception):
    pass
    
class ExperimentSaveNotFoundError(Exception):
    pass

class ThisIsUnholyError(Exception):
    pass


flatten = lambda l: [item for sublist in l for item in sublist]
#--------------------------------------------------------------------------------
#class SubExp(dict):
#    def __init__(self,subexpname,subdirs,models, *args):
#        dict.__init__(self, args)
#        self['name']    =    subexpname
#        self['subdirs'] =    subdirs
#        self['models']  =    models
    
#    def __repr__(self):
#        return('Sub_experience : ' + str(self['name'])+' - Subdirectory : '+str(self['subdirs'])+' - Models ('+str(len(self['models']))+') :'+str(self['models'])[:20]+'...')

#--------------------------------------------------------------------------------
class MetaExperiment:
    
    def __init__(self, exp_norm_fold='add_experiment_folder', experience_name='dummy_exp', rs=23):
        self.init_complete=False            #Attributes related to the tracking of the MetaExperiment object 
        self.method_used_on_library       = []
        self.value_changes_on_library     = []
        self.logs        =                  dict()
        self.temp_value_for_ds_changes =    None
        self.init_complete=True
        
        #Attributes set at the creation or soon after, to select & save data
        self.library =                      None
        #self.save_loc    =                  None
        self.meta_experience_name=          experience_name
        self.meta_experience_path=          None
        self.exp_norm_fold = exp_norm_fold
        
        #Attributes related to the tracking of the MetaExperiment object 
        self.los_pickolos = None
        self.spare_dict=    OrderedDict()
        
        #Attributes generally related to machine-learning tasks
        self.target  =       None
        self.classification= None
        self.train_index, self.test_index= None, None
        self.random_seed       =   rs
        self.train_scorer      =   None
        self.test_scorer       =   None
        self.cross_validate    =   cross_validate
        #Master attribute ?    
        #self.naive_models=   None
        #self.meta_model  =   None
        
        #Attributes related to the first phase of the experiment
        self.phase_1_exp =       None    
        '''Should be a function that produces models in form of a {'lib1_name':lib1_model,...} dict
        Signature of these functions : 
        exp_1_library(DSLibrary),exp_1_targets(pd.DataFrame), self.meta_experience_path(str), 
        self.target(str), self.classification(bool, self.train_scorer(make_scorer(scorer),self.random_seed(int), **kwargs    '''
        self.phase_1_exp_other_args = dict()
        self.phase_1_exp_score =    None
        self.phase_1_exp_retest_score = None
        self.models_dict =   dict()
        self.phase_1_naming_func=make_a_ds_run_name_fp
        
        #Attributes related to the phase 2a of the experiment
        self.phase_2a_exp =         None
        self.phase_2a_exp_retest_score =   None
        
        self.phase_2a_exp_other_args = {'rs':None, 'save':'10CV10_','save_phase':'second_phase_a', 'models':None}
        
        #Attributes related to the phase 4a of the experiment
        self.phase_4a_exp =         None
        self.phase_4a_exp_retest_score =   None
        
        self.phase_4a_exp_other_args = {'rs':None, 'save':'10CV10_','save_phase':'second_phase_a', 'models':None}
        
        #Attributes related to the third phase of the experiment
        self.phase_2b_exp_model_prod =         None
        self.phase_2b_exp_score =   None
        self.phase_2b_exp_retest_score =   None
        self.phase_2b_exp_refined_score =   None
        self.phase_2b_exp_index =   None
        self.phase_2b_models =   None
        self.intermediary_dataset_2b=None
        self.p2b_model_selection=dict()      #models to generate the intermediary datasets !
        self.p2b_intermediaries_models=  dict()
        self.phase_2b_exp_other_args= dict()
        self.phase_2b_naming_func = make_a_ds_run_name_2pb
        self.phase_2b_sub_exp_dict  =  dict()
        
        #Attributes related to the third phase of the experiment
        self.phase_3b_exp =         None
        self.phase_3b_exp_score =   None
        self.phase_2b_exp_refined_score =   None
        self.phase_3b_exp_retest_score =   None
        self.phase_3b_exp_index =   None
        self.phase_3b_exp_other_args = {'rs':None, 'save':'10CV10_','save_phase':'second_phase_b', 'models':None}
        self.intermediary_dataset_3b=None
        
        #Attributes related to the retesting phase of the experiment
        self.phase_2b_exp_retest =         None
        self.phase_2b_exp_args_retest =    None
        self.phase_3b_exp_retest =         None
        self.phase_3b_exp_args_retest =    None
        
        #Attributes related to subbing the third phase of the experiment
        
        self.phase_3b_sub_exp_dict     = list()
        
        
        self.phase_1_dummy_score  = None
        self.phase_2a_dummy_score = None
        self.phase_2b_dummy_score = None
        self.phase_3b_dummy_score = None
        self.phase_4a_dummy_score = None
        self.phase_4b_dummy_score = None
        self.phase_1_dummy_folds  = None
        self.phase_2a_dummy_folds = None
        self.phase_2b_dummy_folds = None
        self.phase_3b_dummy_folds = None
        self.phase_4a_dummy_folds = None
        self.phase_4b_dummy_folds = None
                
        self.meta_experience_path, self.meta_experience_name =  make_a_meta_exp_name(exp_norm_fold,experience_name)
        self.meta_experience_name = experience_name
        self.save_loc             = self.meta_experience_path+'/'+self.meta_experience_name
        
        #Plotting attributes
        self.subdivisions        = dict()
        
    #--------------------------------------------------------------------------------

    def save(self, number=None):
        #Saves the ME instance in it's current state at : 
        #self.meta_experience_path + '/saves/' + self.meta_experience_name+ '_state_'+str(number)+'.pickle'
        
        if not self.meta_experience_path :               raise UndefinedError('No path for ME ? ')
        if not os.path.exists(self.meta_experience_path+'/saves/'):  os.makedirs(self.meta_experience_path+'/saves/')
            
        if not os.path.exists(self.meta_experience_path + '/saves/' + self.meta_experience_name + '_state_0.pickle'):
          save_path = self.meta_experience_path + '/saves/' + self.meta_experience_name + '_state_0.pickle'
        else:
          if not(number):
              allfiles = [f for f in listdir(self.meta_experience_path+'/saves/') if isfile(join(self.meta_experience_path+'/saves/', f))]
              number = max([int(f.split('_')[-1][:-7]) for f in allfiles if ('.pickle' in f and '_state' in f)])+1
          save_path = self.meta_experience_path + '/saves/' + self.meta_experience_name+ '_state_'+str(number)+'.pickle'
        
        with open(save_path,'xb') as infile :
          pickle.dump( self, infile )
        return save_path
    
    @classmethod
    def load_from_pickle(self,experience_folder, meta_experience_name, number = None):
        #Loads the save with highest state in :
        #experience_folder+'/saves/'+ meta_experience_name +'_state_' + str(number) + '.pickle'
        if not os.path.exists(experience_folder+'/saves/'):  raise ExperimentSaveNotFoundError('Experiment does not exist or wasn\'t saved')
        
        allfiles = [f for f in listdir(experience_folder+'/saves/') if isfile(join(experience_folder+'/saves/', f))]
        #problem = Fails when number save >9... CORRECTED !(?)
        if not(number):
            number = max([int(f.split('_')[-1][:-7]) for f in allfiles if ('.pickle' in f and '_state' in f)])
        load_path = experience_folder+'/saves/'+ meta_experience_name +'_state_' + str(number) + '.pickle'
        with open(load_path,'rb') as infile :
          me = pickle.load( infile )
        print('Name of the metaexperience : ',meta_experience_name)
        print('Save number ',str(number), ' loaded.')
        return me
        

    
    
    #--------------------------------------------------------------------------------
    #General manipulation methods for the models_dict attribute, not specifically related to tpot experiments 
    def none_models(self):
        none_list=list()
        for key, value in self.models_dict.items():
           if value is None :
                   none_list.append(key)
        return none_list
    
    def purge_models(self, model_list):
        print('Following models will be taken out (', len(model_list),'in total ) : \n',model_list)
        for model in model_list :
            self.models_dict.pop(model)
    
    def purge_models_of_none(self):
        self.purge_models(self.none_models())
        
    def reset_models(self):
        print('Resetting all models ... ')
        for model in self.models_dict:
            #print(model,' is being resetted ...')
            self.models_dict[model]=clone(self.models_dict[model], safe=True)
    
    #--------------------------------------------------------------------------------
    def switch_name_and_loc(self, exp_norm_fold='add_experiment_folder', experience_name='dummy_exp'):
        self.meta_experience_path, self.meta_experience_name =  make_a_meta_exp_name(exp_norm_fold,experience_name) 
        #self.meta_experience_name =  experience_name
        self.save_loc             = self.meta_experience_path+'/'+self.meta_experience_name
        self.exp_norm_fold = exp_norm_fold
    #--------------------------------------------------------------------------------
    def set_library(self, library):
        #Use this to set the DSLibrary that should be use. Will reset some logs
        self.library = library
        self.method_used_on_library=[]
        self.value_changes_on_library=[]

    def set_target(self, target, classification):
        #Use this to set the DSLibrary that should be use. Will reset some logs
        #Use after matching !
        if type(target) == str :          self.target = target
        else : raise ValueError('\'target\' must be string')
        
        if type(classification)== bool :  self.classification = classification
        else : raise ValueError('\'classification\' must be bool')
        
        if classification : self.dummy_predictors= [DummyClassifier(strategy='most_frequent',random_state=self.random_seed),\
           DummyClassifier(strategy='stratified',random_state=self.random_seed)]
        else : self.dummy_predictors= [DummyRegressor(strategy='mean'), DummyRegressor(strategy='median')]
        
        self.library.target_ds=self.library.inf_ds[[target]]
        
        self.library.inf_ds[target]=None

    def ds_libs_all_indices_to_same(self):
        #Use this to reduce the amount of patient in each DataSet to those present in all DataSet
        self.library=self.library.lib_with_all_shared_patients()
        self.temp_value_for_ds_changes = 'ds_libs_all_indices_to_same'
        
        #--------------------------------------------------------------------------------
        #Initialize the train/test indices in for of pandas.Index objects 

    def init_train_test_indices_random_split_in_two(self,rs=None ):
        if rs is None : rs=self.random_seed
        #randomly splits in two
        kf=KFold(n_splits=2, random_state=self.rs, shuffle=True)
        kf.get_n_splits(self.library.target_ds)
        for x,y in kf.split(self.library.target_ds):
          self.train_index, self.test_index = x,y
        self.test_index=self.library.target_ds.iloc[self.test_index].index
        self.train_index=self.library.target_ds.iloc[self.train_index].index
        self.temp_value_for_ds_changes={'Matching_function':'init_train_test_indices_random_split_in_two'}
        
          
    def init_train_test_indices_matching_splits_in_two(self, parameters=db1, rs=None):
        if rs is None : rs=self.random_seed
        #Make the two datasets have matching indices
        self.train_index, self.test_index = split_in_matching_datasets(self.library.inf_ds, parameters=parameters, rs=rs)
        self.train_index, self.test_index = self.train_index.index, self.test_index.index 
        self.temp_value_for_ds_changes  = {'Matching_function':'init_train_test_indices_matching_splits_in_two', 'matching_param':parameters, 'random_seed':rs}
          
     #--------------------------------------------------------------------------------
    def init_2b_3b_indices_matching_splits_in_two(self, parameters=db1, rs=None):
        if rs is None : rs=self.random_seed
        #Make the two datasets have matching indices
        self.phase_2b_exp_index, self.phase_3b_exp_index = split_in_matching_datasets(self.library.inf_ds.loc[self.test_index], parameters=parameters, rs=rs)
        self.phase_2b_exp_index, self.phase_3b_exp_index = self.phase_2b_exp_index.index, self.phase_3b_exp_index.index 
        self.temp_value_for_ds_changes  = {'Matching_function':'init_2b_3b_indices_matching_splits_in_two', 'matching_param':parameters, 'random_seed':rs}
     
     #--------------------------------------------------------------------------------
    def test_before_exp(self): 
        #run in and before exps
        if self.library is None :                raise UndefinedError('No DSLibrary assigned to experience')
        if self.train_index is None :            raise UndefinedError('No train_index assigned to experience')
        if self.test_index is None :             raise UndefinedError('No test_index assigned to experience')
        if self.save_loc is None :               raise UndefinedError('No save location assigned to experience')
        if self.exp_norm_fold is None :          raise UndefinedError('No normal experience folder location assigned to experience')
        if self.meta_experience_name is None :   raise UndefinedError('No meta-experience name assigned to experience')
        if self.meta_experience_path is None :   raise UndefinedError('No meta-experience path assigned to experience')
        if self.target is None :                 raise UndefinedError('No target assigned to experience')
        if self.classification is None :         raise UndefinedError('No class./reg. assigned to experience')
        if self.random_seed is None :            raise UndefinedError('No random seed assigned to experience')
        if self.train_scorer is None :           raise UndefinedError('No train_scorer assigned to experience')
        if self.test_scorer is None :            raise UndefinedError('No test_scorer assigned to experience')  
        if self.phase_1_exp is None :            raise UndefinedError('No model_producers assigned to experience')    
    
    def create_holy_seal_of_the_first_phase(self):
        #used to make sure some 1 run result don't get destroyed
        if os.path.exists(self.meta_experience_path + '/holy_seal_of_the_first_phase.txt'):
          raise ThisIsUnholyError('The holy seal has already been created it seems !')
          
        else : 
          with open (self.meta_experience_path + '/holy_seal_of_the_first_phase.txt','w') as holyfile:
            holyfile.write('Papapapin')
            print('The seal has been placed')
    
    def what_seal_of_the_first_phase(self):
        #used to remove phase 1 protection results
        if os.path.exists(self.meta_experience_path + '/holy_seal_of_the_first_phase.txt'):
            os.remove(self.meta_experience_path + '/holy_seal_of_the_first_phase.txt')
        else : print('Seal not existent')

    def create_holy_seal_of_the_second_phase_a(self):
        #used to make sure some run-2a-results don't get destroyed
        if os.path.exists(self.meta_experience_path + '/holy_seal_of_the_second_phase_a.txt'):
          raise ThisIsUnholyError('The 2nd (a) holy seal has already been created it seems !')
          
        else : 
          with open (self.meta_experience_path + '/holy_seal_of_the_second_phase_a.txt','w') as holyfile:
            holyfile.write('Papapapin')
            print('The seal has been placed')

    def what_seal_of_the_second_phase_a(self):
        if os.path.exists(self.meta_experience_path + '/holy_seal_of_the_second_phase_a.txt'):
            os.remove(self.meta_experience_path + '/holy_seal_of_the_second_phase_a.txt')
        else : print('Seal not existent')
        
    def create_holy_seal_of_the_second_phase_b(self):
        #used to make sure some run-2b-results don't get destroyed
        if os.path.exists(self.meta_experience_path + '/holy_seal_of_the_second_phase_b.txt'):
          raise ThisIsUnholyError('The 2nd (b) holy seal has already been created it seems !')
          
        else : 
          with open (self.meta_experience_path + '/holy_seal_of_the_second_phase_b.txt','w') as holyfile:
            holyfile.write('Papapapin')
            print('The seal has been placed')
            
    def what_seal_of_the_second_phase_b(self):
        if os.path.exists(self.meta_experience_path + '/holy_seal_of_the_second_phase_b.txt'):
            os.remove(self.meta_experience_path + '/holy_seal_of_the_second_phase_b.txt')
        else : print('Seal not existent')

    def create_holy_seal_of_the_third_phase_b(self):
        #used to make sure some run-2b-results don't get destroyed
        if os.path.exists(self.meta_experience_path + '/holy_seal_of_the_third_phase_b.txt'):
          raise ThisIsUnholyError('The 3rd (b) holy seal has already been created it seems !')
          
        else : 
          with open (self.meta_experience_path + '/holy_seal_of_the_third_phase_b.txt','w') as holyfile:
            holyfile.write('Papapapin')
            print('The seal has been placed')
            
    def what_seal_of_the_third_phase_b(self):
        if os.path.exists(self.meta_experience_path + '/holy_seal_of_the_third_phase_b.txt'):
            os.remove(self.meta_experience_path + '/holy_seal_of_the_third_phase_b.txt')
        else : print('Seal not existent')
    #--------------------------------------------------------------------------------
    
    def load_pickles(self,prefix_model_loc='first_phase', suffix_model_loc='_train_var.pickle'):
        #Sometimes, datas have to be loaded from pickle files.
        for dataset in self.library :
            path_for_pickle = self.meta_experience_path+'/exp_dirs/'+self.library[dataset].ds_name+'/'+prefix_model_loc+'/'+self.library[dataset].ds_name+suffix_model_loc
            with open(path_for_pickle, 'rb')as infile :
              self.los_pickolos[dataset] = pickle.load(infile)
    
    def model_from_los_pickolos(self):
        #Make models in los_pickolos the
        for dataset in self.library :
            self.models_dict[dataset]=copy.deepcopy(self.los_pickolos[dataset]['model'])
            
    def gen_spare_DS_from_lib(self, key_list):
        for key in key_list :
            self.spare_dict[key]=copy.deepcopy(self.library.pop(key))
    
    def put_spare_DS_in_lib(self, key_list=None):
        if key_list is None : key_list  =  copy.deepcopy(list(self.spare_dict.keys()))
        for key in key_list :
            self.library[key]=copy.deepcopy(self.spare_dict.pop(key))
            
            
    #--------------------------------------------------------------------------------    
    def retest_models_on_index_set(self, index,rs=None, save=None, save_phase='first_phase',models=None, subdir='exp_dirs',library=None):
    #Arguments of origin func : self, rs=None,save='10CV10_', save_phase='third_phase_b',models=None,subdir_load='exp_dirs_2',subdir_save='exp_dirs_2',library=None
    #Call to this func : self.phase_3b_exp_index,rs=rs,save=save,save_phase='third_phase_b',models=models,subdir=exp_dirs_2,library=library
        if models is None : models = self.models_dict
        if library is None : library = self.library
        
        if any(self.spare_dict) : 
          print('Warning - Some libraries are in the sparedict - their models won\'t find them !')
          
        if rs is None : rs=self.random_seed
        libs_missing_models, not_all_libs_retest = list(), False
        
        for key in library:
        #Checking if a library has to be retest are not    
            if key not in models : 
              not_all_libs_retest = True
              libs_missing_models.append(key)
            else :
              if models[key] is None : 
                not_all_libs_retest = True
                raise MissingModelError(str(key+' model does not have an associated model (Value is None).'))
              
        if not_all_libs_retest :
          print('Datasets that will not be retested :', libs_missing_models)
          print('Retest models will be : ',list(models.keys()))
          x=input('Sure ?[y/N]')
          if x!='y': raise MissingModelError(str('Following libraries are missing in models dictionnary :', libs_missing_models))


        np.random.seed(rs)
        scores = dict()
        all_folds=dict()
        retest_library = library.lib_with_same_indices(index)
        retest_targets = library.target_ds.loc[index][self.target]
        print('Phase being 10 times 10 Fold c-ved : ', save_phase)
        z=1
        for dataset in models :
            X_orig = retest_library[dataset].df_data
            y_orig = retest_targets
            scores[library[dataset].ds_name]=list()
            #scores[self.library[dataset].ds_name] = cross_validate(self.models_dict[dataset], X, y,scoring=self.test_scorer, cv=10)
            print('Tested dataset : ',dataset,' - ',z,' / ',len(models))
            z+=1
            for i in range(10):
                np.random.seed(rs+i)
                X, y = shuffle(X_orig, y_orig, random_state=rs+i)
                print('\tIteration ',str(i),' out of 10')
                try:
                    current_score, current_folds= self.cross_validate(models[dataset], X, y,scoring=self.test_scorer, cv=10, rs=rs+i)
                    scores[library[dataset].ds_name].append(deepcopy(current_score))
                    all_folds['Time '+str(i)]=deepcopy(current_folds)
                except:
                    scores[library[dataset].ds_name].append(np.NaN)
                    print('\tWARNING : Failure in iteration ',i,' - ', dataset)

            if save is not None : 
              if not os.path.exists(self.meta_experience_path + '/'+subdir+'/' + dataset +'/'+save_phase):\
                os.makedirs(self.meta_experience_path + '/'+subdir+'/' + dataset +'/'+save_phase+'/')
              X_orig.to_csv(self.meta_experience_path + '/'+subdir+'/' + dataset +'/'+save_phase+'/features_'+dataset+'_retest.csv')
              y_orig.to_csv(self.meta_experience_path + '/'+subdir+'/' + dataset +'/'+save_phase+'/target_'+dataset+'_retest.csv')
              save_path =   self.meta_experience_path + '/'+subdir+'/' + dataset + '/'+ save_phase +'/'+save+dataset+'_retest.pickle'
              #pd.DataFrame(all_folds, index=all_folds.keys(), columns=all_folds['Time 0'].keys()).to_csv(self.meta_experience_path + '/'+subdir+'/' + dataset + '/'+ save_phase +'/'+save+dataset+'_folds.csv')
              pd.DataFrame.from_dict(all_folds).to_csv(self.meta_experience_path + '/'+subdir+'/' + dataset + '/'+ save_phase +'/'+save+dataset+'_folds.csv')
              print('Writing to : ',str(self.meta_experience_path + '/'+subdir+'/' + dataset + '/'+ save_phase +'/'+save+dataset))
              with open(save_path,'wb') as infile :
                pickle.dump( scores[library[dataset].ds_name], infile )
              
              save_path =   self.meta_experience_path + '/'+subdir+'/' + dataset + '/'+ save_phase +'/'+save+dataset+'_retest.csv'
              try:
                pd.DataFrame(scores[library[dataset].ds_name]).to_csv(save_path)
              except :
                print('Could not write dataset : ',dataset,' - score as CSV (pickled available though')
            
                
        return scores
        #x.phase_1_dummy_score[list(x.phase_1_dummy_score.keys())[0]]['test_MAE'].std()
    
    def display_logs(self):
        for i in (self.logs.keys()):
            print('At ',i[0],' change on :', i[1])
    #--------------------------------------------------------------------------------
    def not_retested_models(self,save_phase, save='10CV10_'):
        investigated_models=list()
        for directory in os.listdir(self.meta_experience_path+'/exp_dirs/'):
            if not os.path.exists(self.meta_experience_path+'/exp_dirs/'+directory+'/'+save_phase+'/'+save+directory+'_retest.pickle'):
                investigated_models.append(directory)
        print('Following models will be tested again (',str(len(investigated_models)),') :',investigated_models)
        return investigated_models

    def retest_train_set(self,rs=None,save='10CV10_',save_phase='first_phase',models=None):
        if rs is None : rs=self.random_seed
        self.phase_1_exp_retest_score=self.retest_models_on_index_set(self.train_index, rs, save=save, save_phase=save_phase,models=models)
        #self.save()
    
    def retest_test_set(self,rs=None,save='10CV10_',save_phase='second_phase_a',models=None):
        if rs is None : rs=self.random_seed
        self.phase_2a_exp_retest_score=self.retest_models_on_index_set(self.test_index, rs, save=save, save_phase=save_phase,models=models)
        #self.save()
        
    def retest_second_b_set(self,rs=None,save='10CV10_',save_phase='second_phase_b',models=None,subdir='exp_dirs',library=None):
        if rs is None : rs=self.random_seed
        self.phase_2b_exp_retest_score=self.retest_models_on_index_set(self.phase_2b_exp_index, rs, save=save, save_phase=save_phase,models=models,subdir=subdir,library=library)
        #self.save()
           
    def retest_third_b_set(self,rs=None, save='10CV10_',save_phase='third_phase_b', models=None,subdir='exp_dirs',library=None):
        if rs is None : rs=self.random_seed
        self.phase_3b_exp_retest_score=self.retest_models_on_index_set(self.phase_3b_exp_index, rs, save=save, save_phase=save_phase, models=models,subdir=subdir,library=library)
        #self.save()
        
    def refined_second_b_set_test(self,rs=None,save='10CV10_',save_phase='second_phase_b',models=None,subdir='exp_dirs_2',library=None):
        if rs is None : rs=self.random_seed
        if models is None : models= self.p2b_intermediaries_models
        if library is None : library= self.intermediary_dataset_2b
        
        self.phase_2b_exp_refined_score=self.retest_models_on_index_set(self.phase_2b_exp_index, rs, save=save, save_phase=save_phase,models=models,subdir=subdir,library=library)
        #self.save()
           
    def refined_third_b_set_test(self,rs=None, save='10CV10_',save_phase='third_phase_b', models=None,subdir='exp_dirs_2',library=None):
        if rs is None : rs=self.random_seed
        if models is None : models= self.p2b_intermediaries_models
        if library is None : library= self.intermediary_dataset_3b
        
        self.phase_3b_exp_refined_score=self.retest_models_on_index_set(self.phase_3b_exp_index, rs, save=save, save_phase=save_phase, models=models,subdir=subdir,library=library)
        #self.save()
    
    def retest_fourth_a_set(self,rs=None,save='10CV10_',save_phase='fourth_phase_a',models=None):
        if rs is None : rs=self.random_seed
        self.phase_4a_exp_retest_score=self.retest_models_on_index_set(self.library[list(self.library.keys())[0]].df_data.index, rs, save=save, save_phase=save_phase,models=models)
        #self.save()
    
    def restart_retest_train_set(self,rs=None, save='10CV10_', save_phase='first_phase', models=None):
        if models is None :
          models=self.not_retested_models(save_phase, save)
          models=dict([(i,self.models_dict[i]) for i in models]) #converting to a dict of models
        self.retest_train_set(rs=rs, save=save,save_phase=save_phase,models=models)       

    def restart_retest_test_set(self,rs=None, save='10CV10_', save_phase='second_phase_a', models=None):
        if models is None :
          models=self.not_retested_models(save_phase, save)
          models=dict([(i,self.models_dict[i]) for i in models]) #converting to a dict of models
        self.retest_test_set(rs=rs, save=save,save_phase=save_phase,models=models)
    
    #--------------------------------------------------------------------------------
    def return_score_from_retest(self,save_phase, save='10CV10_', models = None, subdir='exp_dirs'):
        if models is None : models = self.models_dict.keys()
        scores=dict()
        investigated_models=list()
        for directory in os.listdir(self.meta_experience_path+'/'+subdir+'/'):
          if directory in models :
            if os.path.exists(self.meta_experience_path+'/'+subdir+'/'+directory+'/'+save_phase+'/'+save+directory+'_retest.pickle'):
                with open(self.meta_experience_path+'/'+subdir+'/'+directory+'/'+save_phase+'/'+save+directory+'_retest.pickle','rb') as infile :
                  scores[directory]= pickle.load( infile )
            else :
                print(self.meta_experience_path+'/'+subdir+'/'+directory+'/'+save_phase+'/'+save,directory+'_retest.pickle'+' was not found')
        return scores
    
    
    def scan_results_for_nan_and_struct(self, score_o):
      score = copy.deepcopy(score_o)
      result = dict()
      structure = None
      #while structure is None :
      
      for ds in score :
        for results in score[ds]: 
          if results is not np.NaN :
              structure=dict()
              for keys in results:
                structure[keys]=None
              break
        if structure is not None : break
              
      print('Structure seems to be :', structure)
      for ds in score :
        for i, result in enumerate(score[ds]):
          if type(result)==float : 
            if math.isnan(result) :
              print('NaN value spotted')
              score[ds][i]=copy.deepcopy(structure)
      return score, structure
          
    
    def generate_simplified_results(self, score, to_extract='test_MAE' ):
      result = dict()
      for ds in score :
        #print(ds,' - results : ',score[ds])
        result[ds]=dict() 
        result[ds][to_extract+'_mean'] = np.mean(flatten(pd.DataFrame(score[ds])[to_extract].dropna()))
        #print('Mean :',score[ds])
        result[ds][to_extract+'_std']  = np.std(flatten(pd.DataFrame(score[ds])[to_extract].dropna()))
        #print('Std :',score[ds])
      return pd.DataFrame(result)
      #x=new_ageme.display_score_from_retest(save_phase='first_phase')
      #pd.DataFrame(z).transpose().sort_values('mean')
    
    def return_occurences(self, structure, searching='test_'):
      scorers = list()
      for key in structure :
          if searching in key :
              scorers.append(key)
      return scorers
    
    def results_refined(self,save_phase, save='10CV10_',subdir='exp_dirs', models = None):
      scores = self.return_score_from_retest(save_phase, save=save , models = models,subdir=subdir)
      scores, structure = self.scan_results_for_nan_and_struct(scores)
      temp_df = None
      for scorer in self.return_occurences(structure):
          print('Looking for : ',scorer)
          if temp_df is None :
              temp_df = pd.DataFrame(self.generate_simplified_results(scores, to_extract=scorer ))
          else :
              temp_df = temp_df.append(self.generate_simplified_results(scores, to_extract=scorer ))
      return temp_df.transpose().sort_values(self.return_occurences(structure)[0]+'_mean', ascending=False)
    
    def save_results_refined(self,save_phase, save='10CV10_', models = None, adder='_score_save', subdir='exp_dirs'):
      if not os.path.exists(self.meta_experience_path+'/mew_results/'+save_phase+'/'):
        os.makedirs(self.meta_experience_path+'/mew_results/'+save_phase+'/')
      self.results_refined(save_phase, save=save, models = models,subdir=subdir).to_csv(\
      self.meta_experience_path+'/mew_results/'+save_phase+'/'+self.meta_experience_name+'_'+save+save_phase+adder+'.csv')
      
#    def save_results_refined(self,save_phase, save='10CV10_', models = None, adder='_score_save',save=False):
#      if not os.path.exists(self.meta_experience_path+'/mew_results/'+save_phase+'/'):
#        os.makedirs(self.meta_experience_path+'/mew_results/'+save_phase+'/')
#      results_refined = self.results_refined(save_phase, save=save, models = models)
#      if save :results_refined.to_csv(\
#        self.meta_experience_path+'/mew_results/'+save_phase+'/'+self.meta_experience_name+'_'+save+save_phase+adder+'.csv')
#      return results_refined
    
    #--------------------------------------------------------------------------------
    def dummy_experiment_on_set(self, index,rs=None,save='dummy_',save_phase=None):
        if rs is None : rs=self.random_seed
        scores = dict()
        np.random.seed(rs)
        
        X_orig = self.library[list(self.library.keys())[0]].df_data.loc[index]
        y_orig = self.library.target_ds.loc[index]
        absolutely_all_folds=dict()
                    
        for predictor in self.dummy_predictors :
            scores[str(predictor)]=list()
            #scores[str(predictor)] = cross_validate(predictor, X, y,scoring=self.test_scorer, cv=10)
            all_folds=dict()
            for i in range(10):
                np.random.seed(rs+i)
                X, y = shuffle(X_orig, y_orig, random_state=rs+i)
                results, current_folds = self.cross_validate(predictor, X, y,scoring=self.test_scorer, cv=10, rs=rs+i)
                scores[str(predictor)].append(results)
                all_folds['Time '+str(i)]=deepcopy(current_folds)
            absolutely_all_folds[str(predictor)]=deepcopy(pd.DataFrame.from_dict(all_folds))
            pd.DataFrame.from_dict(all_folds).to_csv(self.meta_experience_path + '/'+'mew_results'+'/'+ save_phase +'/'+save+str(predictor)[:30]+'_folds.csv')
        return copy.deepcopy(scores), copy.deepcopy(absolutely_all_folds)
    
    '''
      rs=23
      scores=dict()
      index=exp_anr.train_index
      save='dummy_'
      save_phase='second_phase_a'
      X_orig = exp_anr.library[list(exp_anr.library.keys())[0]].df_data.loc[index]
      y_orig = exp_anr.library.target_ds.loc[index]
      absolutely_all_folds=dict()
      for predictor in exp_anr.dummy_predictors :
        scores[str(predictor)]=list()
        all_folds=dict()
        #scores[str(predictor)] = cross_validate(predictor, X, y,scoring=exp_anr.test_scorer, cv=10)
        for i in range(10):
            np.random.seed(rs+i)
            X, y = shuffle(X_orig, y_orig, random_state=rs+i)
            results, current_folds = exp_anr.cross_validate(predictor, X, y,scoring=exp_anr.test_scorer, cv=10, rs=rs+i)
            scores[str(predictor)].append(results)
            all_folds['Time '+str(i)]=deepcopy(current_folds)
        pd.DataFrame.from_dict(all_folds).to_csv(exp_anr.meta_experience_path + '/'+'mew_results'+'/'+ save_phase +'/'+save+'_test_folds.csv')
    '''
    def dummy_experiment_train_set(self,rs=None,save='dummy_',save_phase='first_phase'):
        if rs is None : rs=self.random_seed
        self.phase_1_dummy_score, self.phase_1_dummy_folds =self.dummy_experiment_on_set(self.train_index, rs,save=save,save_phase=save_phase)

    def dummy_experiment_2a_set(self,rs=None,save='dummy_',save_phase='second_phase_a'):
        if rs is None : rs=self.random_seed
        self.phase_2a_dummy_score, self.phase_2a_dummy_folds =self.dummy_experiment_on_set(self.test_index,rs,save=save,save_phase=save_phase)
        
        #x.phase_1_dummy_score[list(x.phase_1_dummy_score.keys())[0]]['test_MAE'].std()
    
    def dummy_experiment_2b_set(self,rs=None,save='dummy_',save_phase='second_phase_b'):
        if rs is None : rs=self.random_seed
        self.phase_2b_dummy_score, self.phase_2b_dummy_folds =self.dummy_experiment_on_set(self.phase_2b_exp_index,rs,save=save,save_phase=save_phase)

    def dummy_experiment_3b_set(self,rs=None,save='dummy_',save_phase='third_phase_b'):
        if rs is None : rs=self.random_seed
        self.phase_3b_dummy_score, self.phase_3b_dummy_folds =self.dummy_experiment_on_set(self.phase_3b_exp_index,rs,save=save,save_phase=save_phase)
    def dummy_experiment_4a_set(self,rs=None,save='dummy_',save_phase='fourth_phase_a'):
        if rs is None : rs=self.random_seed
        self.phase_4a_dummy_score, self.phase_4a_dummy_folds =self.dummy_experiment_on_set(self.library[list(self.library.keys())[0]].df_data.index,rs,save=save,save_phase=save_phase)
    
    def display_a_score(self, score, score_type):
        for key in score:
            results = []
            [results.extend(i[score_type]) for i in score[key]]
            print(key,' : ', np.mean(results),' +/- ',np.std(results),' - ',len(results),' iterations used.')
    
    def return_simplified_score(self, score, score_type):
        results_dict=dict()
        for key in score:
            results = []
            [results.extend(i[score_type]) for i in score[key]]
            print(key,' : ', np.mean(results),' +/- ',np.std(results),' - ',len(results),' iterations used.')
            results_dict[key]={'average':np.mean(results), 'std':np.std(results)}
        return pd.DataFrame(results_dict).transpose()
        #return {'average':np.mean(results), 'std':np.std(results)}
                
    def save_dummy_results(self, raw_score,save='10CV10_',save_phase='second_phase_b',adder='_dummy_score_save'):
        scores, structure = self.scan_results_for_nan_and_struct(raw_score)
        temp_df = None
        for scorer in self.return_occurences(structure):
            print('Looking for : ',scorer)
            if temp_df is None :
                temp_df = pd.DataFrame(self.generate_simplified_results(scores, to_extract=scorer ))
            else :
                temp_df = temp_df.append(self.generate_simplified_results(scores, to_extract=scorer ))
        if not os.path.exists(self.meta_experience_path+'/mew_results/'+save_phase+'/'):
          os.makedirs(self.meta_experience_path+'/mew_results/'+save_phase+'/')
        temp_df.transpose().sort_values(self.return_occurences(structure)[0]+'_mean', ascending=False).\
          to_csv(self.meta_experience_path+'/mew_results/'+save_phase+'/'+self.meta_experience_name+'_'+save+save_phase+adder+'.csv')
    
    def run_all_dummies_on_exp_dir(self, rs=None):
        self.dummy_experiment_train_set(rs=rs)
        self.dummy_experiment_2a_set(rs=rs)
        self.dummy_experiment_2b_set(rs=rs)
        self.dummy_experiment_3b_set(rs=rs)
        self.dummy_experiment_4a_set(rs=rs)
        self.save_dummy_results(self.phase_1_dummy_score,save='10CV10_',save_phase='first_phase',adder='_dummy_score_save')
        self.save_dummy_results(self.phase_2a_dummy_score,save='10CV10_',save_phase='second_phase_a',adder='_dummy_score_save')
        self.save_dummy_results(self.phase_2a_dummy_score,save='10CV10_',save_phase='second_phase_b',adder='_dummy_score_save')
        self.save_dummy_results(self.phase_3b_dummy_score,save='10CV10_',save_phase='third_phase_b',adder='_dummy_score_save')
        self.save_dummy_results(self.phase_4a_dummy_score,save='10CV10_',save_phase='fourth_phase_a',adder='_dummy_score_save')

    def gather_all_retest_raw(self, with_save=False):
        #self.dummy_experiment_train_set(rs=rs)
        #self.dummy_experiment_2a_set(rs=rs)
        #self.dummy_experiment_2b_set(rs=rs)
        #self.dummy_experiment_3b_set(rs=rs)
        #return_score_from_retest(self,save_phase, save='10CV10_', models = None, subdir='exp_dirs')
        self.phase_1_exp_retest_score   = self.return_score_from_retest(save='10CV10_',save_phase='first_phase',    models=None)
        self.phase_2a_exp_retest_score = self.return_score_from_retest(save='10CV10_',save_phase='second_phase_a', models=None)
        self.phase_2b_exp_retest_score = self.return_score_from_retest(save='10CV10_',save_phase='second_phase_b', models=None)
        self.phase_3b_exp_retest_score = self.return_score_from_retest(save='10CV10_',save_phase='third_phase_b',  models=None)
        self.phase_4a_exp_retest_score = self.return_score_from_retest(save='10CV10_',save_phase='fourth_phase_a',  models=None)
        
        if with_save :
          self.save_results_refined(save_phase='first_phase', save='10CV10_', models = None, adder='_score_save', subdir='exp_dirs')
          self.save_results_refined(save_phase='second_phase_a', save='10CV10_', models = None, adder='_score_save', subdir='exp_dirs')
          self.save_results_refined(save_phase='second_phase_b', save='10CV10_', models = None, adder='_score_save', subdir='exp_dirs')
          self.save_results_refined(save_phase='third_phase_b', save='10CV10_', models = None, adder='_score_save', subdir='exp_dirs')
          self.save_results_refined(save_phase='fourth_phase_a', save='10CV10_', models = None, adder='_score_save', subdir='exp_dirs')
        
    def gather_all_retest_refined(self, with_save=False):
        #self.dummy_experiment_train_set(rs=rs)
        #self.dummy_experiment_2a_set(rs=rs)
        #self.dummy_experiment_2b_set(rs=rs)
        #self.dummy_experiment_3b_set(rs=rs)
        #return_score_from_retest(self,save_phase, save='10CV10_', models = None, subdir='exp_dirs')
        self.phase_2_b_exp_refined_score = self.return_score_from_retest(save='10CV10_',save_phase='second_phase_b',    models=self.p2b_intermediaries_models, subdir = 'exp_dirs_2')
        self.phase_3_b_exp_refined_score = self.return_score_from_retest(save='10CV10_',save_phase='third_phase_b',     models=self.p2b_intermediaries_models, subdir = 'exp_dirs_2')
        
        if with_save :
          self.save_results_refined(save_phase='second_phase_b', save='10CV10_', models = self.p2b_intermediaries_models, adder='_refined_score_save', subdir='exp_dirs_2')
          self.save_results_refined(save_phase='third_phase_b', save='10CV10_', models = self.p2b_intermediaries_models, adder='_refined_score_save', subdir='exp_dirs_2')
        
    #--------------------------------------------------------------------------------
    def first_phase(self,rs=None):
        if rs is None : rs=self.random_seed
        np.random.seed(rs)
        
        self.test_before_exp()
        self.create_holy_seal_of_the_first_phase()
        
        exp_1_library = self.library.lib_with_same_indices(self.train_index)
        exp_1_targets = self.library.target_ds.loc[self.train_index]
        
        self.models_dict = self.phase_1_exp(exp_1_library,exp_1_targets,\
           self.meta_experience_path, self.target, self.classification,\
           self.train_scorer,self.random_seed, self.phase_1_exp_other_args, self.phase_1_naming_func)
        #ds_library, exp_path='experiment', target='ds', classification=True,scorer=None,rand=23, gen=5

    def second_phase_a(self, rs=None):
        if rs is None : rs=self.random_seed
        np.random.seed(rs)
        
        self.test_before_exp()
        self.create_holy_seal_of_the_second_phase_a()
        
        #exp_2a_library = self.library.lib_with_same_indices(self.test_index)
        #exp_2a_targets = self.library.target_ds.loc[self.test_index]
        
        self.phase_2a_exp_retest_score = self.retest_models_on_index_set(self.test_index)
        self.save()

#    def meta_experiment_same_indices(self, first_pass, second_func, third_func,rs=23):
#        self.models_dict  = first_pass(self)
    
    def restart_exp(self):
        #Attention ! self.uncomplete_models must have a value here that is the list of the exp names which have not been completed and should be restarted
        print('ATTENTION! Use with caution, as this might delete important files')
        self.what_seal_of_the_first_phase()
        for exp in self.uncomplete_models:
            if os.path.isdir(self.meta_experience_path+'/exp_dirs/'+exp+'/first_phase/'):
                print(self.meta_experience_path+'/exp_dirs/'+exp+'/first_phase/ is about to be removed. Type \"y\" to confirm.')
                x=input('Sure ?[y/N]')
                if x=='y':  shutil.rmtree(self.meta_experience_path+'/exp_dirs/'+exp+'/first_phase/')
                else:('Folder preserved - might be overwritten in future steps !')
        self.gen_spare_DS_from_lib(self.library.keys()-self.uncomplete_models)
        x=''
        print('Restart about to be executed on :',self.library.keys())
        x=input('Sure ?[y/N]')
        if x=='y':  self.first_phase()
        else:('Restarting aborted !')
    #--------------------------------------------------------------------------------
    def generate_sub_exps(self, library=None, models_dict=None, split_ds_by={'All':['_'],'Connectivity':['df_cod'],'Functional_CM':['df_fcm'],'Structural':['_na']},
                            drop_ds={'Connectivity':['df_cod'],'Functional_CM':['df_fcm'],'Structural':['df_na'],'All':['']}):
        if library     is None : library = self.library
        if models_dict is None : models_dict = self.models_dict
        
        sub_exps=list()
        for key in split_ds_by:
          new_models_dict=dict() 
          for new_key in library.keys():
            for string in split_ds_by[key]:
              if string in new_key :
                 if new_key not in drop_ds[key] : new_models_dict[new_key]=clone(models_dict[new_key])
                 break
          sub_exps.append((key,new_models_dict))
          #print(key,' (',len(new_keys),') : ',new_keys)
          
        return dict(sub_exps)
    
    def models_selector_for_2b(self, models=None,rs=None, std=[None,0,0.5,1], greater_is_better=True,save='10CV10_', sort_by=None):
      #Checks on different variables
      self.test_before_exp()
      if rs is None : rs=self.random_seed
      if models is None : models = list(self.models_dict.keys())
      if any(self.spare_dict) : raise DictsInSpareDictError('Some libraries are in the sparedict - their models won\'t find them !')
      for key in self.library:
              if key not in self.models_dict : raise MissingModelError(str(key+' missing in models_dict'))
              if self.models_dict[key] is None : raise MissingModelError(str(key+' value in models_dict is None'))
      
      #Initializations
      np.random.seed(rs)
      self.reset_models()
      if self.phase_1_dummy_score is None : self.dummy_experiment_train_set(rs=rs)
      self.dummy_experiment_2b_set(rs=rs)
      self.dummy_experiment_3b_set(rs=rs)
  
      #Choosing score to select with from score structure (should be homogeneous among scores)
      _,structure = self.scan_results_for_nan_and_struct(self.phase_1_dummy_score)
      if sort_by is None :
          for key in structure :
              if 'test_' in key :
                  scorer = key
                  break
      else : scorer=sort_by
  
      #Extracting results on train_set for comparison
      results = self.results_refined('first_phase', save=save, models = models)
      #Extracting dummy_values (average and std) for comparisons
      dummy_values = self.return_simplified_score(self.phase_1_dummy_score, scorer)
      dummy_values_p2b = self.return_simplified_score(self.phase_2b_dummy_score, scorer)
      
      if std is not None:
          for value in std :
              if value is None :
                  self.p2b_model_selection[str(value)]=copy.deepcopy(self.models_dict)
              else :
                  if greater_is_better :
                      best_dummy = dummy_values.sort_values('average',ascending=False).iloc[0]
                      added_value = best_dummy['std']*value
                      threshold   = added_value + best_dummy['average']
                      temp_dict = copy.deepcopy([(model, self.models_dict[model]) for model in results.loc[results[scorer+'_mean']>=threshold].index])
                      self.p2b_model_selection[str(value)]=dict()
                      for key, model in temp_dict:
                        self.p2b_model_selection[str(value)][key]=model
                  else : raise
    
    def model_selector_for_subexps(self, models_dict_subdict, generate_for=['None','1'], origin_model_dict=None):
    #Goal is to provide a subselection of dataset for 2b and 3b.
    #Could help provide comparison between original dataset and splitted datasets ensembled
    #Can also help for graphic interface
    #use after model selection and with sub-dict
        if origin_model_dict is None : origin_model_dict = self.p2b_model_selection 
        for x in  generate_for :
          if x not in self.p2b_model_selection : raise 
        #print('heeeey')
        new_sub_dict=dict()
        for subdict in models_dict_subdict:
          for origin_dict in generate_for :
              new_sub_dict[subdict+'_'+origin_dict]=dict()
              for key in models_dict_subdict[subdict] :
                  if key in origin_model_dict[origin_dict] :
                      new_sub_dict[subdict+'_'+origin_dict][key]= models_dict_subdict[subdict][key]
        to_drop = []
        for key in new_sub_dict :
          if not new_sub_dict[key]:
            to_drop.append(key)
        for key in to_drop :
          new_sub_dict.pop(key)
        
        return new_sub_dict
    
    def updatemodel_selection(self, new_model_dict):
        new_d = dict
        for d in (self.p2b_model_selection, new_model_dict): new_d.update(d)
        return new_d
    
    def create_estimation_datasets(self, rs=None):
        
        if rs is None : rs=self.random_seed
        exp_1_library = self.library.lib_with_same_indices(self.train_index)
        exp_1_targets = self.library.target_ds.loc[self.train_index]
        
        exp_2b_library = self.library.lib_with_same_indices(self.phase_2b_exp_index)
    #    exp_2b_targets = self.library.target_ds.loc[self.phase_2b_exp_index]
        
        exp_3b_library = self.library.lib_with_same_indices(self.phase_3b_exp_index)
        
        self.intermediary_dataset_2b= DSLibrary('2b DS library')
        self.intermediary_dataset_2b.inf_ds    = copy.deepcopy(self.library.inf_ds)
        self.intermediary_dataset_2b.target_ds = copy.deepcopy(self.library.target_ds)
        temp_data_2b = dict()
        
        self.intermediary_dataset_3b= DSLibrary('3b DS library')
        self.intermediary_dataset_3b.inf_ds    = copy.deepcopy(self.library.inf_ds)
        self.intermediary_dataset_3b.target_ds = copy.deepcopy(self.library.target_ds)
        temp_data_3b = dict()
        
        for std in self.p2b_model_selection:
            print(std,' models are in usage ...')
            temp_data_2b[std] = pd.DataFrame(index=self.phase_2b_exp_index)
            temp_data_3b[std] = pd.DataFrame(index=self.phase_3b_exp_index)
            for model in self.p2b_model_selection[std]:
                np.random.seed(rs)
                print(model,' is being used')
                self.p2b_model_selection[std][model].fit(exp_1_library[model].df_data, exp_1_targets)
                temp_data_2b[std][model] = self.p2b_model_selection[std][model].predict(exp_2b_library[model].df_data)
                temp_data_3b[std][model] = self.p2b_model_selection[std][model].predict(exp_3b_library[model].df_data)
            
        for std in self.p2b_model_selection:
            DataSet(temp_data_2b[std], str(std),dataset_lib=self.intermediary_dataset_2b, description=std)
            DataSet(temp_data_3b[std], str(std),dataset_lib=self.intermediary_dataset_3b, description=std)
    
    def second_phase_b_preparation(self):
        self.p2b_model_selection = self.model_selector_for_subexps(self.generate_sub_exps(),origin_model_dict=self.models_selector_for_2b())
        self.create_estimation_datasets()
    #--------------------------------------------------------------------------------
    def second_phase_b(self, rs=None,just_a_string=False, subdir='exp_dirs_2', library=None):
        #new_func = produce_the_namer(self.meta_experience_path,subdir=subdir)
        
        if rs is None : rs=self.random_seed
        np.random.seed(rs)
        self.test_before_exp()
        self.create_holy_seal_of_the_second_phase_b()       
        
        if library is None : exp_2b_library = self.intermediary_dataset_2b
        else : exp_2b_library=library
        exp_2b_targets = self.intermediary_dataset_2b.target_ds.loc[self.phase_2b_exp_index]
        
        for lib in exp_2b_library :
            if exp_2b_library[lib].df_data.shape[0]==0 : raise ValueError('Empty ds in 2b intermediary_dataset_2b')

        if just_a_string : str(str(exp_2b_library)+'\nTarget : '+str(exp_2b_targets)[:30]+' ...\nExperience path :'+\
           self.meta_experience_path+'\nTarget : '+ self.target+'\nIs a classification : '+ str(self.classification),'\nTraining scorer : '+\
           str(self.train_scorer)+'\nRandom seed : '+str(self.random_seed)+'\nOther argument dict : '+ str(self.phase_2b_exp_other_args)+'\nNaming function :'+ str(self.phase_2b_naming_func))
           
        else : self.phase_2b_models = self.phase_2b_exp_model_prod(exp_2b_library,exp_2b_targets,\
           self.meta_experience_path, self.target, self.classification,\
           self.train_scorer,self.random_seed, self.phase_2b_exp_other_args, self.phase_2b_naming_func)
           
           #self.models_dict = self.phase_1_exp(exp_1_library,exp_1_targets,\
           #self.meta_experience_path, self.target, self.classification,\
           #self.train_scorer,self.random_seed, self.phase_1_exp_other_args, self.phase_1_naming_func)
    
    # new_ageme.restart_exp_2b(function=new_ageme.second_phase_b,rs=23,uncomplete_models=['0'])
    #--------------------------------------------------------------------------------
    def restart_exp_2b(self, subdir = 'exp_dirs_2',phase='second_phase_b', function=None,rs = None, unsealing_func=None, uncomplete_models=None,library=None):
        if function is None : function = self.second_phase_b
        if library is None : library = copy.deepcopy(self.intermediary_dataset_2b)
        if unsealing_func is None : unsealing_func=self.what_seal_of_the_second_phase_b
        if uncomplete_models is None : raise
        #Attention ! self.uncomplete_models must have a value here that is the list of the exp names which have not been completed and should be restarted
        print('ATTENTION! Use with caution, as this might delete important files')
        unsealing_func()
        to_dump = library.keys()-uncomplete_models
        for key in to_dump :
            library.pop(key)
        for exp in uncomplete_models:
            if os.path.isdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+phase+'/'):
                print(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+phase+' is about to be removed. Type \"y\" to confirm.')
                x=input('Sure ?[y/N]')
                if x=='y':  shutil.rmtree(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+phase+'/')
                else:('Folder preserved - might be overwritten in future steps !')
            else :
                print(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+phase+' not existent')
        x=''
        print('Restart about to be executed on :',library.keys())
        x=input('Sure ?[y/N]')
        if x=='y':  function(rs=rs,subdir=subdir, library=library)
        else:('Restarting aborted !')
    
    def salvage_gen_models(self, subdir='exp_dirs_2',phase='second_phase_b'):
        #Lets recover models from tpot runs
        #return list of libs name for which mature model could not be found
        temp_model_dict = dict()
        if not os.path.exists(self.meta_experience_path+'/'+subdir):  raise ExperimentSaveNotFoundError('Experiment does not exist or wasn\'t saved')
        
        all_dirs=os.listdir(self.meta_experience_path+'/'+subdir)
        uncomplete_models=list()
        for exp in all_dirs :
            print('\nLoading experiment models : ',exp,end='')
            if not os.path.isfile(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+phase+'/'+exp+'_train_var.pickle'):
                  print(' - WARNING : FAILURE Model in early stage of generation - Can not be loaded.', end='')
                  uncomplete_models.append(exp)
                  temp_model_dict[exp]= None
            else : 
                with open(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+phase+'/'+exp+'_train_var.pickle','rb') as infile :
                    results = pickle.load( infile )
                    temp_model_dict[exp]=clone(results['model'])
                print(' - SUCCESS : Model Loaded from pickle', end='')
        print('')
        print('Unfinished models are considered to be (',len(uncomplete_models),'):',uncomplete_models)
        return temp_model_dict, uncomplete_models
    
    #retest_models_on_index_set(self, index,rs=None, save=None, save_phase='first_phase',models=None, subdir='exp_dirs',library=None)
    

    def third_phase_b(self, rs=None,save='10CV10_', save_phase='third_phase_b',models=None,subdir_load='exp_dirs_2',subdir_save='exp_dirs_2',library=None):
        if rs is None : rs=self.random_seed
        np.random.seed(rs)
        
        self.test_before_exp()
        self.create_holy_seal_of_the_third_phase_b()
        uncomplete_models=None
        if models is None : 
          if self.p2b_intermediaries_models is None :
            self.p2b_intermediaries_models,uncomplete_models=self.salvage_gen_models('exp_dirs_2','second_phase_b')
          models = self.p2b_intermediaries_models
          print('Uncomplete models : ',uncomplete_models)
          #models, uncomplete_models=self.salvage_gen_models(subdir=subdir_load)
        if library is None :library = copy.deepcopy(self.intermediary_dataset_3b)
        
        keys_to_throw = library.keys()-models.keys()
        for key in keys_to_throw:
                library.pop(key)
        #exp_2a_library = self.library.lib_with_same_indices(self.test_index)
        #exp_2a_targets = self.library.target_ds.loc[self.test_index]
        
        self.phase_3b_exp_score = self.retest_models_on_index_set(self.phase_3b_exp_index,rs=rs,save=save,save_phase=save_phase,models=models,subdir=subdir_save,library=library)
        self.save()
    
    def from_int_models_to_mew_results(self, save_phase='third_phase_b', save='10CV10_', models = None, adder='_dummy_score_save', subdir='exp_dirs_2'):
        if models is None : 
            if self.p2b_intermediaries_models is None :
              self.p2b_intermediaries_models,uncomplete_models=self.salvage_gen_models('exp_dirs_2','second_phase_b')
            else : uncomplete_models='Not computed - models already attached'
            models = self.p2b_intermediaries_models
            print('Uncomplete models : ',uncomplete_models)
        self.save_results_refined(save_phase,save=save,subdir=subdir,models=models, adder='_refined_score_save')
        self.dummy_experiment_3b_set()
        self.dummy_experiment_2b_set()
        #self.save_results_refined(self,save_phase, save='10CV10_', models = None, adder='_score_save', subdir='exp_dirs')
        
        scores, structure = self.scan_results_for_nan_and_struct(self.phase_3b_dummy_score)
        temp_df = None
        for scorer in self.return_occurences(structure):
            print('Looking for : ',scorer)
            if temp_df is None :
                temp_df = pd.DataFrame(self.generate_simplified_results(scores, to_extract=scorer ))
            else :
                temp_df = temp_df.append(self.generate_simplified_results(scores, to_extract=scorer ))
        if not os.path.exists(self.meta_experience_path+'/mew_results/'+save_phase+'/'):
          os.makedirs(self.meta_experience_path+'/mew_results/'+save_phase+'/')
        temp_df.transpose().sort_values(self.return_occurences(structure)[0]+'_mean', ascending=False).\
          to_csv(self.meta_experience_path+'/mew_results/'+save_phase+'/'+self.meta_experience_name+'_'+save+save_phase+adder+'.csv')
    #retest_second_b_set()
    #retest_third_b_set()  
    
      
    
    #--------------------------------------------------------------------------------
    # Generall functions ! If everything was set correctly, they should suffice, combined with phase launching functions.
    
    '''
    How it goes, with correct init (see children classes for examples)
    1. : Correct initialization ( Further descriptions coming soon ...)
    
    2. : Compute dummy experiments for each phase 
    
      self.run_all_dummies_on_exp_dir()     
    
    3. : Run phase 1
    
      self.phase_1_exp()
    
    4. : Run all raw retests
    
      self.launch_all_raw_retest()
      
    4.b: Run second_phase_a if this phase differs from a simple retest
    
      self.second_phase_a()
    
    4.c: Gather retest results (in case of reinit for example)
    
      self.gather_all_retest_raw(with_save=True)
    
    5. : Prepare second b Sub-libraries !
      self.second_phase_b_preparation()
      Note : Arguments can not be passed for now. Comming soon ...
    
    6. : Launch phase 2b
    
    7. Gather models in exp.p2b_intermediary_models
      -> in exp.p2b_intermediary_models=exp.salvage_tpot_models(subir='exp_dirs_2')[0]
        -> Make sure evey lib has a model (might be necessary to lower gen threshold)
    
    8. : Launch phase 3b
    #Error potential: p2b_intermediary_models empty dict
    
    9.  launch_all_b_retest()
        (refined_second_b_set_test()
        refined_third_b_set_test())
    
    10. gather_all_retest_refined(with_save=True)
    
    '''
    
    def launch_all_raw_retest(self):
      #requires phase 1 to have have ended successfully
      self.retest_train_set()
      self.retest_test_set()
      self.retest_second_b_set()
      self.retest_third_b_set()
      self.retest_fourth_a_set()
    
    def launch_all_b_retest(self):
      #requires phase 2b to have ended successfully
      self.refined_second_b_set_test()
      self.refined_third_b_set_test()

    def retests_and_scores_for_refined(self):
      #Launch this after launch_all_b_retest
      self.from_int_models_to_mew_results()
      self.from_int_models_to_mew_results(save_phase='second_phase_b')
    
    
    #--------------------------------------------------------------------------------
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
    
    
    def generate_one_result_df(self, score_o, to_extract='test_MAE' ):
      #Generating boxplots requires all results of a 10 times 10 Fold to be in a single array
      #This functions takes a aci-kit learn score and returns as one array the results of all the runs as a DataFrame
      result = dict()
      score,_ = self.scan_results_for_nan_and_struct(score_o)
      for ds in score :
        #print(ds,' - results : ',score[ds]) 
        result[ds] = flatten(pd.DataFrame(score[ds])[to_extract].dropna())
      
      max_length = max([len(x) for x in result.values()])
      
      for ds in result :
        if len(result[ds]) < max_length :
            needed_iterations  = max_length - len(result[ds])
            list_to_append     = [np.nan]*needed_iterations
            result[ds] = result[ds] + list_to_append
            #print(ds,' new length : ', len(result[ds]))
      
      return pd.DataFrame(result)
      
    def boxplot_from_score_with_dummy(self,df, best_dummy_score, dir_path=None, file_name = 'test_file',columns=None,title='', score_legend=''):
      #This function draws a boxplot using a scikit score flattened according to 'generate one result'
      #It also takes a dummy score modified the same way
      #And generates a nice plots saved under 'path' 
      
      plt.clf()
      df['Random Predictor'] = best_dummy_score
      columns=list(columns)
      columns.append('Random Predictor')
      boxplot = df[columns].boxplot(rot=45,fontsize=10, grid=False)
      
      fig     = boxplot.get_figure()
      axes    = fig.get_axes()
      
      dummy_line_middle = axes[0].axhline(df['Random Predictor'].quantile(0.5),color='grey',ls='-')
      dummy_line_middle.set_label('Median random prediction')
      
      dummy_line_middle_plus = axes[0].axhline(df['Random Predictor'].quantile(0.75),color='grey',ls='--')
      dummy_line_middle_plus.set_label('Q3 random prediction')
      
      dummy_line_middle_minus = axes[0].axhline(df['Random Predictor'].quantile(0.25),color='grey',ls='--')
      dummy_line_middle_minus.set_label('Q1 random prediction')
      
      fig.set_size_inches(12.5, 7.25)
      
      plt.legend()
      plt.ylabel(score_legend)
      plt.title(title)
      plt.tight_layout()
      path = self.save_without_deleting(dir_path, file_name = file_name, extension = '.png')
      plt.savefig(path)
      
    def barplot_from_score_with_dummy(self,df, best_dummy_score, dir_path=None, file_name = 'test_file',columns=None,title='', score_legend=''):
      #This function draws a barplot using a scikit score flattened according to 'generate one result'
      #It also takes a dummy score modified the same way
      #And generates a nice plots saved under 'path' 
      
      plt.clf()
      df['Random Predictor'] = best_dummy_score
      columns=list(columns)
      columns.append('Random Predictor')
      #print(df)
      #print('Df columns:',df.columns)
      #print('Df index:', df.index)
      #barplot = df[columns].mean().abs().plot.bar(yerr=[df.mean().abs()-df.std(), df.mean().abs()+df.std()],rot=45,fontsize=10, grid=False)
      barplot = df[columns].mean().abs().plot.bar(yerr=[df[columns].std()],rot=45,fontsize=10, grid=False)
      #plt.bar(np.arange(df.shape[1]), df.mean(), yerr=[df.mean()-df.min(), df.max()-df.mean()], capsize=6)
      fig     = barplot.get_figure()
      axes    = fig.get_axes()
      '''
      dummy_line_middle = axes[0].axhline(abs(df['Random Predictor'].quantile(0.5)),color='grey',ls='-')
      dummy_line_middle.set_label('Median random prediction')
      
      dummy_line_middle_plus = axes[0].axhline(abs(df['Random Predictor'].quantile(0.75)),color='grey',ls='--')
      dummy_line_middle_plus.set_label('Q3 random prediction')
      
      dummy_line_middle_minus = axes[0].axhline(abs(df['Random Predictor'].quantile(0.25)),color='grey',ls='--')
      dummy_line_middle_minus.set_label('Q1 random prediction')
      '''
      fig.set_size_inches(12.5, 7.25)
      
      #plt.legend(loc='upper left')
      plt.ylabel(score_legend)
      plt.title(title)
      plt.tight_layout()
      path = self.save_without_deleting(dir_path, file_name = file_name, extension = '.png')
      plt.savefig(path)
    
    def table_plot_for_overfitting(self,df_result_train, df_result_test, dir_path=None, file_name = 'test_file_tp',title='Measure of overfitting',top=None):
      #This function draws a boxplot using a scikit score flattened according to 'generate one result'
      #It also takes a dummy score modified the same way
      #And generates a nice plots saved under 'path' 
      if top is None : top = df_result_train.shape[0]
      
      plt.clf()
      
      table = pd.concat([df_result_train, df_result_test],axis=1).iloc[:top]
      table.rename(columns={ table.columns[0]: "Train Dataset Result" }, inplace = True)
      table.rename(columns={ table.columns[1]: "Test Dataset Result" }, inplace = True)
      table['Difference']  =    table["Test Dataset Result"]-table["Train Dataset Result"]
      
      #plt.figure()

      # table
      #plt.subplot(121)
      #plt.plot(table)
      cell_text = []
      for row in range(len(table)):
          cell_text.append(table.iloc[row])
      
      plt.table(cellText=cell_text, colLabels=table.columns, rowLabels=table.index)
      #plt.table(cellText=cell_text, colLabels=table.columns, rowLabels=table.index, loc='center')
      plt.axis('off')

      #plt.tight_layout()
      #plt.legend()
      #plt.title(title)
      #file_name = '_my_picture', extension = '.png'
      path = self.save_without_deleting(dir_path, file_name = file_name, extension = '.png')
      plt.savefig(path, pad_inches=1)
           

    def draw_top_x_boxplots(self, score, associated_dummy_score, dir_path=None, file_name='test_name',x=10,title='', to_extract='test_MAE', score_legend='', from_these=None,library=None):
      #Takes the scikit score for a run and the associated dummy, reduces them to 1D arrays and sends them to 
      #boxplot_from_score_with_dummy for drawing.
      
      results            =   self.generate_one_result_df(score_o=score,to_extract=to_extract)
      dummy_values       =   self.return_simplified_score(associated_dummy_score, to_extract)
      best_dummy_name    =   dummy_values.sort_values('average',ascending=False).iloc[0].name
      best_dummy_score   =   self.generate_one_result_df(score_o=associated_dummy_score, to_extract=to_extract)[best_dummy_name]
      columns            =   results.mean()
      if from_these is not None : columns            =   columns[from_these].sort_values(ascending=False).index[:x]
      else :                      columns            =   columns.sort_values(ascending=False).index[:x]
      
      new_index_dict = dict()
      for lib in library :
        new_index_dict[lib]=library[lib].description
      results = results.rename(columns=new_index_dict)
      new_columns = [new_index_dict[i] for i in columns]
      self.boxplot_from_score_with_dummy(df=results,best_dummy_score=best_dummy_score, dir_path=dir_path, file_name=file_name, columns=new_columns, title=title,  score_legend=score_legend)

    def draw_top_x_barplots(self, score, associated_dummy_score, dir_path=None, file_name='test_name',x=10,title='', to_extract='test_MAE', score_legend='', from_these=None,library=None):
      #Takes the scikit score for a run and the associated dummy, reduces them to 1D arrays and sends them to 
      #boxplot_from_score_with_dummy for drawing.
      
      results            =   self.generate_one_result_df(score_o=score,to_extract=to_extract)
      dummy_values       =   self.return_simplified_score(associated_dummy_score, to_extract)
      best_dummy_name    =   dummy_values.sort_values('average',ascending=False).iloc[0].name
      best_dummy_score   =   self.generate_one_result_df(score_o=associated_dummy_score, to_extract=to_extract)[best_dummy_name]
      columns            =   results.mean()
      if from_these is not None : columns            =   columns[from_these].sort_values(ascending=False).index[:x]
      else :                      columns            =   columns.sort_values(ascending=False).index[:x]
      
      new_index_dict = dict()
      for lib in library :
        new_index_dict[lib]=library[lib].description
      results = results.rename(columns=new_index_dict)
      new_columns = [new_index_dict[i] for i in columns]
      self.barplot_from_score_with_dummy(df=results,best_dummy_score=best_dummy_score, dir_path=dir_path, file_name=file_name, columns=new_columns, title=title,  score_legend=score_legend)

#generate_sub_exps
    
    def draw_boxplots_for_these(self, score, associated_dummy_score, dir_path=None, file_name='test_name',
                x=10,title='', to_extract='test_MAE', score_legend='', from_these=None,
                split_ds_by={'All':['_'],'Connectivity':['df_cod'],'Functional_CM':['df_fcm'],'Structural':['_na']},
                drop_ds={'Connectivity':['df_cod'],'Functional_CM':['df_fcm'],'Structural':['df_na'],'All':['']},library=None):
      
      subexps = self.generate_sub_exps(split_ds_by=split_ds_by, drop_ds=drop_ds)
      for subexp in subexps :
        self.draw_top_x_boxplots(score=score, 
                        associated_dummy_score=associated_dummy_score, 
                        dir_path=dir_path, file_name = file_name+'_'+subexp,x=x,
                        title=subexp+' '+title, to_extract=to_extract,  score_legend=score_legend,
                        from_these = list(subexps[subexp].keys()), library=library)
        
    def boxplots_for_refined_phases(self, x=10, to_extract='test_MAE', plot_func = None, title='',  score_legend='',library=None):
      #The two most meaningful phases for plottingg are, for now, phases 3b and 4a.
      #This function draws the plots associated to them.
      
      if plot_func is None : plot_func = self.draw_top_x_boxplots
      if library   is None : library   = self.intermediary_dataset_3b
      path_3b = self.meta_experience_path+'/plots/'+'third_phase_b'+'/'
      
      plot_func(score=self.phase_3b_exp_refined_score, associated_dummy_score=self.phase_3b_dummy_score, dir_path=path_3b, file_name = '3_b_boxplot_10_best_refined',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)

    def draw_barplots_for_these(self, score, associated_dummy_score, dir_path=None, file_name='test_name',
                x=10,title='', to_extract='test_MAE', score_legend='', from_these=None,
                split_ds_by={'All':['_'],'Connectivity':['df_cod'],'Functional_CM':['df_fcm'],'Structural':['_na']},
                drop_ds={'Connectivity':['df_cod'],'Functional_CM':['df_fcm'],'Structural':['df_na'],'All':['']},library=None):
      
      subexps = self.generate_sub_exps(split_ds_by=split_ds_by, drop_ds=drop_ds)
      for subexp in subexps :
        self.draw_top_x_barplots(score=score, 
                        associated_dummy_score=associated_dummy_score, 
                        dir_path=dir_path, file_name = file_name+'_'+subexp,x=x,
                        title=subexp+' '+title, to_extract=to_extract,  score_legend=score_legend,
                        from_these = list(subexps[subexp].keys()), library=library)
        
    def barplots_for_refined_phases(self, x=10, to_extract='test_MAE', plot_func = None, title='',  score_legend='',library=None):
      #The two most meaningful phases for plottingg are, for now, phases 3b and 4a.
      #This function draws the plots associated to them.
      
      if plot_func is None : plot_func = self.draw_top_x_barplots
      if library   is None : library   = self.intermediary_dataset_3b
      path_3b = self.meta_experience_path+'/plots/'+'third_phase_b'+'/'
      
      plot_func(score=self.phase_3b_exp_refined_score, associated_dummy_score=self.phase_3b_dummy_score, dir_path=path_3b, file_name = '3_b_barplot_10_best_refined',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)

    def repartition_histogramms_by_class(self, dir_path=None, file_name='test', title='', var = 'Age',class_to_dif='Gender',df=None):
      #Histogramms of the df. Differentiate by class_to_dif
      #This function draws the plots and saves them in path.
      
      plt.clf()
      #fig = plt.figure()
      #hist = df.groupby(class_to_dif).hist()
      print(df)
      print(class_to_dif)
      hist = sns.pairplot(df, hue=class_to_dif, kind='reg', diag_kind='hist',size=2.5, vars=list(df.drop(class_to_dif,axis=1)))
      #sns.histplot(class_to_dif, var, data=df, palette=["lightblue", "lightgreen"]).set_title(title)
      #hist.fig.suptitle(title,  y=1.8)
      #fig     = hist[0].get_figure()
      #axes    = fig.get_axes()
      #fig.set_size_inches(12.5, 7.25)
      #plt.tight_layout()
      path = self.save_without_deleting(dir_path, file_name = file_name, extension = '.png')
      plt.savefig(path)
    
    def hist_for_age_and_gender(self, var='Age',class_to_dif='Gender', drop=[]):
      #draw repartition plots of the general information dataset
      org_df           = pd.concat([self.library.inf_ds.dropna(axis=1),self.library.target_ds],axis=1)
      org_df           = org_df.drop(drop,axis=1)
      df_1,  path_1    = org_df.loc[self.train_index],self.meta_experience_path+'/plots/'+'first_phase'+'/'
      df_2a, path_2a   = org_df.loc[self.test_index], self.meta_experience_path+'/plots/'+'second_phase_a'+'/'
      df_4a, path_4a   = org_df, self.meta_experience_path+'/plots/'+'fourth_phase_a'+'/'
      
      self.repartition_histogramms_by_class(dir_path=path_1, file_name='hist_of_rep_1', title='Repartition of train set', var=var,class_to_dif=class_to_dif,df=df_1)
      self.repartition_histogramms_by_class(dir_path=path_2a, file_name='hist_of_rep_2a', title='Repartition of test set', var=var, class_to_dif=class_to_dif,df=df_2a)
      self.repartition_histogramms_by_class(dir_path=path_4a, file_name='hist_of_rep_4a', title='Repartition of whole set', var=var, class_to_dif=class_to_dif,df=df_4a)
      
    def boxplots_for_test_phases_raw(self, x=10, to_extract='test_MAE', plot_func = None, title='',  score_legend='',library=None):
      #The two most meaningful phases for plotting are, for now, phases 3b and 4a.
      #This function draws the plots associated to them.
      
      if plot_func is None : plot_func = self.draw_top_x_boxplots
      if library   is None : library   = self.library
      path_2a = self.meta_experience_path+'/plots/'+'second_phase_a'+'/'
      path_3b_raw = self.meta_experience_path+'/plots/'+'third_phase_b'+'/'
      path_4a = self.meta_experience_path+'/plots/'+'fourth_phase_a'+'/'
      
      plot_func(score=self.phase_2a_exp_retest_score, associated_dummy_score=self.phase_2a_dummy_score, dir_path=path_2a, file_name = '2_a_boxplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)
      plot_func(score=self.phase_3b_exp_retest_score, associated_dummy_score=self.phase_3b_dummy_score, dir_path=path_3b_raw, file_name = '3_b_boxplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)
      plot_func(score=self.phase_4a_exp_retest_score, associated_dummy_score=self.phase_4a_dummy_score, dir_path=path_4a, file_name = '4_a_boxplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)      
      
    def barplots_for_test_phases_raw(self, x=10, to_extract='test_MAE', plot_func = None, title='',  score_legend='',library=None):
      #The two most meaningful phases for plotting are, for now, phases 3b and 4a.
      #This function draws the plots associated to them.
      
      if plot_func is None : plot_func = self.draw_top_x_barplots
      if library   is None : library   = self.library
      path_2a = self.meta_experience_path+'/plots/'+'second_phase_a'+'/'
      path_3b_raw = self.meta_experience_path+'/plots/'+'third_phase_b'+'/'
      path_4a = self.meta_experience_path+'/plots/'+'fourth_phase_a'+'/'
      
      plot_func(score=self.phase_2a_exp_retest_score, associated_dummy_score=self.phase_2a_dummy_score, dir_path=path_2a, file_name = '2_a_barplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)
      plot_func(score=self.phase_3b_exp_retest_score, associated_dummy_score=self.phase_3b_dummy_score, dir_path=path_3b_raw, file_name = '3_b_barplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)
      plot_func(score=self.phase_4a_exp_retest_score, associated_dummy_score=self.phase_4a_dummy_score, dir_path=path_4a, file_name = '4_a_barplot_10_best_raw',x=x,title=title, to_extract=to_extract,  score_legend=score_legend, library=library)
      
    def table_plot_overfit_measurement(self, to_extract='test_MAE'):
      #Two datasets are used to search for models. To measure overfit, we compare their results with fully independent datasets
      phase_1_df    = self.results_refined('first_phase'   , save='10CV10_',subdir='exp_dirs', models = None)[[to_extract+'_mean']].sort_values(by=to_extract+'_mean', ascending=False)
      phase_1_df.rename(columns={ phase_1_df.columns[0]: "Other_name" }, inplace = True) 
      #\-> Names gotta be different between the two dataframes send to the plotting function
      
      phase_2a_df   = self.results_refined('second_phase_a', save='10CV10_',subdir='exp_dirs', models = None)[[to_extract+'_mean']].loc[phase_1_df.index]
      self.table_plot_for_overfitting(phase_1_df, phase_2a_df, dir_path=self.meta_experience_path+'/plots/'+'first_phase'+'/')
      
      phase_2b_df   = self.results_refined('second_phase_b', save='10CV10_',subdir='exp_dirs_2', models = self.p2b_intermediaries_models)[[to_extract+'_mean']].sort_values(by=to_extract+'_mean', ascending=False)
      phase_2b_df.rename(columns={ phase_2b_df.columns[0]: "Other_name" }, inplace = True)
      #\-> Names gotta be different between the two dataframes send to the plotting function
      
      phase_3b_df   = self.results_refined('third_phase_b' , save='10CV10_',subdir='exp_dirs_2', models = self.p2b_intermediaries_models)[[to_extract+'_mean']].loc[phase_2b_df.index]
      self.table_plot_for_overfitting(phase_2b_df,phase_3b_df, dir_path=self.meta_experience_path+'/plots/'+'second_phase_a'+'/')

# Example : my_me.boxplots_for_test_phases_raw(plot_func= my_me.draw_boxplots_for_these, title = '- data based 10 best model\'s score for Age prediction', to_extract='test_MAE',score_legend='neg. mean absolute error')

    #--------------------------------------------------------------------------------
    def other_set_attribute(self, name, value):
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
        
        if name=='library':
          self.value_changes_on_library.append({timestampStr:value})
          
        if name=='temp_value_for_ds_changes':
          self.method_used_on_library.append({timestampStr:value})
        self.logs[(timestampStr,name)]=value
        self.__dict__[name] = value

    def other_representation(self):
        try :
          self.test_before_exp()
        except UndefinedError:
          print('WARNING ! Some values are not defined')
          
        return('Meta-experience name : ' + str(self.meta_experience_name)+'\n'+
               (str(self.library).split('\n'))[0]+'\n'+
              'Target : '+ str(self.target)+'\n'+
              'Train/test indices : '+ str(self.train_index)[:20]+' ... '+'/'+ str(self.test_index)[:20]+' ...'+'\n'+
              'Saved_under : '+ str(self.meta_experience_path)+'\n'+
              'Saving mask : ' + str(self.save_loc))
        
    def __setattr__(self, name, value):
        try :
          if not (self.init_complete):
            self.__dict__[name] = value
            return None
        except AttributeError :
            self.__dict__[name] = value
            return None
            
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")         
        
        if name=='library':
          self.value_changes_on_library.append({timestampStr:value})
          
        if name=='temp_value_for_ds_changes':
          self.method_used_on_library.append({timestampStr:value})
        self.logs[(timestampStr,name)]=value
        self.__dict__[name] = value
        
    def __repr__(self):
        try :
          self.test_before_exp()
        except UndefinedError:
          print('WARNING ! Some values are not defined')
          
        return('Meta-experience name : ' + str(self.meta_experience_name)+'\n'+
        
               (str(self.library).split('\n'))[0]+'\n'+
              'Target : '+ str(self.target)+'\n'+
              'Train/test indices : '+ str(self.train_index)[:20]+' ... '+'/'+ str(self.test_index)[:20]+' ...'+'\n'+
              'Saved_under : '+ str(self.meta_experience_path)+'\n'+
              'Saving mask : ' + str(self.save_loc))
