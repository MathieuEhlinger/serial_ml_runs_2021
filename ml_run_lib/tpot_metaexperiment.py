# -*- coding: utf-8 -*-
"""
@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""
from ml_run_lib.metaexperiment import MetaExperiment
import os 
import numpy as np
import pandas as pd
import pickle
import ml_run_lib.model_producers
from ml_run_lib.experiments_functions_new import make_a_meta_exp_name, make_a_ds_run_name_fp
from ml_run_lib.tpot_code_reader import tpot_code_execution, tpot_model_score

class ExperimentSaveNotFoundError(Exception):
    pass

class TPOTMetaExperiment(MetaExperiment):
    #--------------------------------------------------------------------------------
    #General manipulation methods for the models_dict attribute, specifically related to tpot experiments
    #Modified to return a dict of complete models and a list of uncomplete models, allowing usage in phase 2 b
    def __init__(self, exp_norm_fold='add_experiment_folder', experience_name='dummy_exp',rs=23):
        super().__init__(exp_norm_fold=exp_norm_fold, experience_name=experience_name,rs=rs)
        self.phase_1_exp             = ml_run_lib.model_producers.multithreading_tpot
        self.phase_2b_exp_model_prod = ml_run_lib.model_producers.multithreading_tpot

    def salvage_tpot_models(self, gen_threshold=5,suffix='_model_temp', subdir = 'exp_dirs', current_phase='first_phase',verbosity=1):
        #Lets recover models from tpot runs
        #return list of libs name for which mature model could not be found
        if not os.path.exists(self.meta_experience_path+'/'+subdir):  raise ExperimentSaveNotFoundError('Experiment does not exist or wasn\'t saved')
        
        all_dirs=os.listdir(self.meta_experience_path+'/'+subdir)
        loaded_models, uncomplete_models=dict(), list()
        for exp in all_dirs :
            if verbosity > 0 : print('\nLoading experiment models : ',exp,end='')
            if not os.path.isfile(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+'_train_var.pickle'):
                if verbosity >1 : print('\nNo pickled model was found ...')
                if os.path.isfile(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+'_final_model.py'):
                    print('Loading at :',self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+'_final_model.py')
                    loaded_models[exp]= tpot_code_execution(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+'_final_model.py',verbosity=verbosity)
                    print(' - SUCCESS : Model Loaded from final_model.py', end='')
                elif os.path.isdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix):
                      if verbosity >1 : print('No final_model.py was found ... Searching for temporary dir.')
                      if verbosity >1 : print('Being searched at : ', self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix)
                      highest_gen = \
                      max([i.split('_')[2] for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix)])
                      
                      highest_idx = \
                      max([i.split('_')[4] for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix) \
                      if i.split('_')[2]==highest_gen ])
                      
                      file_to_load= \
                      [i for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix)\
                      if i.split('_')[2]==highest_gen and i.split('_')[4]==highest_idx][0]
                      
                      loaded_models[exp]= tpot_code_execution(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix+'/'+file_to_load)
                      if int(highest_gen)<gen_threshold : 
                          uncomplete_models.append(exp)
                          if verbosity > 0 : print(' - CAUTION : Model Loaded from generation ',highest_gen,', index :',highest_idx,'model.py', end='')
                      else : 
                          if verbosity >0 :print(' - SUCCESS : Model Loaded from generation ',highest_gen,', index :',highest_idx,'model.py', end='')
                else : 
                  if verbosity > 0 : print(' - WARNING : FAILURE Model in early stage of generation - Can not be loaded.', end='')
                  uncomplete_models.append(exp)
                  loaded_models[exp]= None
                  
            else : 
                with open(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+'_train_var.pickle','rb') as infile :
                    print('Current file :',exp)
                    results = pickle.load( infile )
                    loaded_models[exp]=results['model']
                print(' - SUCCESS : Model Loaded from pickle', end='')
        print('')
        #uncomplete_models.extend(list(self.library.keys()-self.models_dict.keys()))
        print('Unfinished models are considered to be (',len(uncomplete_models),'):', uncomplete_models)
        return loaded_models , uncomplete_models
    
    def find_highest_generation(self,suffix='model_temp', subdir='exp_dirs', current_phase='first_phase'):
        if not os.path.exists(self.meta_experience_path+'/'+subdir):  raise ExperimentSaveNotFoundError('Experiment does not exist or wasn\'t saved')
        all_dirs=os.listdir(self.meta_experience_path+'/'+subdir)
        results=dict()
        
        for exp in all_dirs :
          print('Loading experiment results : ',exp)
          if os.path.isdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix):
                      highest_gen = \
                      max([i.split('_')[2] for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix)])
                      
                      highest_idx = \
                      max([i.split('_')[4] for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix) \
                      if i.split('_')[2]==highest_gen ])
                      
                      file_to_load= \
                      [i for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix)\
                      if i.split('_')[2]==highest_gen and i.split('_')[4]==highest_idx][0]
                      score = tpot_model_score(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix+'/'+file_to_load)
                      results[exp]={'Generation':highest_gen,'Index':highest_idx,'CV Result':np.float(score)}
                      print(exp,' CV Results : ', float(score),' at gen :',int(highest_gen),' (idx:',int(highest_idx),')')
          else :
                      results[exp]={'Generation':np.NaN,'Index':np.NaN,'CV Result':np.NaN}
                      print(exp,'( No results found!')
        if not os.path.exists(self.meta_experience_path+'/results/tpot_run'): os.makedirs(self.meta_experience_path+'/results/tpot_run')        
        pd.DataFrame(results).transpose().to_csv(self.meta_experience_path+'/results/tpot_run/'+current_phase+'tpot_run_results.csv')
        return (pd.DataFrame(results).transpose())

    def list_generations_as_df(self,suffix='model_temp', subdir = 'exp_dirs', current_phase='first_phase'):
        if not os.path.exists(self.meta_experience_path+'/'+subdir):  raise ExperimentSaveNotFoundError('Experiment does not exist or wasn\'t saved')
        all_dirs=os.listdir(self.meta_experience_path+'/'+subdir)
        results=dict()
        
        for exp in all_dirs :
          print('Loading experiment results : ',exp)
          if os.path.isdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix):
                      highest_gen = \
                      max([i.split('_')[2] for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix)])
                      
                      highest_idx = \
                      max([i.split('_')[4] for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix) \
                      if i.split('_')[2]==highest_gen ])
                      
                      list_gen = \
                      [i.split('_')[2] for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix)]
                      
                      list_idx = \
                      [i.split('_')[4] for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix)]
                      
                      score=dict()
                      for j,_ in enumerate(list_gen):
                        file_to_load= \
                        [i for i in os.listdir(self.meta_experience_path+'/'+subdir+'/'+exp+'/'+current_phase+'/'+exp+suffix)\
                            if i.split('_')[2]==list_gen[j] and i.split('_')[4]==list_idx[j]][0]
                        score[str(list_gen[j])+'.'+str(list_idx[j])] = np.float(tpot_model_score(self.meta_experience_path+'/'+subdir+'/'+exp+\
                            '/'+current_phase+'/'+exp+suffix+'/'+file_to_load))
                      score['highest_generation']=np.float(str(highest_gen)+'.'+str(highest_idx))
                      results[exp]=score
          else :
                      results[exp]={}
                      print(exp,'( No results found!')
        if not os.path.exists(self.meta_experience_path+'/results/tpot_run'): os.makedirs(self.meta_experience_path+'/results/tpot_run')        
        pd.DataFrame(results).transpose().drop('highest_generation',axis=1).to_csv(self.meta_experience_path+'/results/tpot_run/'+current_phase+'_list_tpot_run_models.csv')
        with_stats=pd.DataFrame(results).transpose()
        with_stats['Growth']=with_stats.drop('highest_generation',axis=1).max(axis=1)-with_stats.drop('highest_generation',axis=1).min(axis=1)
        with_stats.to_csv(self.meta_experience_path+'/results/tpot_run/'+current_phase+'list_tpot_run_models_stats.csv')
        return (pd.DataFrame(results).transpose())
    
    def monitoring_tpot(self,suffix='_model_temp', subdir='exp_dirs', current_phase='first_phase'):
      o=self.find_highest_generation(suffix, subdir, current_phase).dropna()
      o['CV Result']=o['CV Result'].apply(float)
      o['Generation']=o['Generation'].apply(float)
      return {'Finished':o.loc[o['Generation']>=5], 'Unfinished':o.loc[o['Generation']<5],'Best scores':o.sort_values('CV Result')}
      #exp.monitoring_tpot(subdir='exp_dirs_2', current_phase='second_phase_b')
      
    def split_score_tpot_by_dataset(self, str_for_df_type={'anatomy':'_na','fcm':'fcm','connectivity':'cod'}, suffix='model_temp', subdir='exp_dirs', current_phase='first_phase'):
      #Allows to measure fir
      tpot_res = self.find_highest_generation(current_phase=current_phase)
      for key in str_for_df_type :
        tpot_res.loc[[i for i in tpot_res.index if str_for_df_type[key] in i]].describe().to_csv(self.meta_experience_path+'/results/tpot_run/tpot_'+current_phase+'_'+key+'_describe.csv')
        tpot_res.loc[[i for i in tpot_res.index if str_for_df_type[key] in i]].to_csv(self.meta_experience_path+'/results/tpot_run/tpot_'+current_phase+'_'+key+'_data.csv')
    
