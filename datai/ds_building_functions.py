#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:09:51 2019

@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""

from scipy import integrate


import numpy as np
import pandas as pd
import copy
#from scorer import kappa
import os,sys
#

#
#get_cwd = os.getcwd()
#os.chdir(os.path.dirname(__file__))
#
##import pkgutil
#search_path = ['.'] # set to None to see all modules importable from sys.path
#all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
#print(all_modules)
#

from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from datai.preprocessing import data_preprocessing

from datai.ds_filters import ds_4CAT, ds_cs_b, ds_2CAT,ds_2CAT_b, gender_to_b

j_parcellation='add_j_parcellation_file'
func_folder = 'add_func_folder'
general_inf_file = 'add_general_info_file'
our_data_folder='add_data_folder'
private_data_folder='add_private_data_folder'

if not(os.path.isdir(our_data_folder)):
    os.mkdir(our_data_folder)

jp_csv=pd.read_csv(j_parcellation, header=None)
default_ds_lib=dict()

def check_for_index_in_list(index, indices):
    #returns True if an index is present in a list of index
    is_present=False
    for u in indices:
        if index.equals(u):       is_present=True
    return is_present

def position_of_index_in_list(index, indices):
    #returns position of index in a list of index
    is_present,i=False,0
    for u in indices:
        if index.equals(u):       
            is_present=i
            return i
        i+=1
    return None

class DataSet():
    #DataSet method built around the pd.DataFrame class
    #Adds some possibilities for description and registering in a more general dict
    build_method = None
    ds_name = None
    df_data= None
    description = None
 
    def __init__(self, df_or_path, ds_name, description='', build_method=None, dataset_lib=default_ds_lib):
        ###########################################################
        self.ds_name = copy.deepcopy(ds_name)
        
        if type(df_or_path)==str:  self.df_data = pd.read_csv(df_or_path+'/'+ds_name+'.csv',index_col=0).dropna()
        elif type(df_or_path)==pd.DataFrame: self.df_data = copy.deepcopy(df_or_path)
        else : raise
        
        self.df_data.index = self.df_data.index.astype(int)

        dataset_lib[ds_name] = self
        self.description = copy.deepcopy(description)
    
class DSLibrary(OrderedDict):
    #Used to store banks of DS and run tests on them

    def __init__(self, name='default name'):
        self.library_name=name
        self.inf_ds=None
        self.target_ds=None
    
    
    def all_intersections(self):
        #returns an index corresponding to the indices shared accross all DataSets in the DSLib
        shared_keys=copy.deepcopy(self.target_ds.index)
        inc = 0
        for i in self.keys():
            shared_keys = pd.Index.intersection(shared_keys, self[i].df_data.index)
            inc+=1
        shared_keys = pd.Index.intersection(shared_keys, self.target_ds.index)
        shared_keys = pd.Index.intersection(shared_keys, self.inf_ds.index)
        print('Number of analyzed datasets :', inc+2)
        return shared_keys
    
    def lib_with_same_indices(self, indices):
        #returns a copy of all libs with index reduced to those given in indices, incl. the information DS
        new_library=DSLibrary() 
        for key in self:
            new_library[key]=copy.deepcopy(self[key])
            new_library[key].df_data=new_library[key].df_data.loc[indices]
        new_library.target_ds =self.target_ds.loc[indices]
        new_library.inf_ds =self.inf_ds.loc[indices]
        return new_library

    def lib_with_all_shared_patients(self):
        #returns a copy of all the libs with index reduced to subjects with ID shared in all DataSets
        return self.lib_with_same_indices(self.all_intersections())
    
    def lib_by_list_of_ds(self, list_of_ds,name=''):
        #returns a copy of the library with only the mentionned DataSets
        new_library=DSLibrary(name)
        new_library.target_ds  =  copy.deepcopy(self.target_ds)
        new_library.inf_ds     =  self.inf_ds
        
        for ds in list(list_of_ds):
          new_library[ds]=copy.deepcopy(self[ds])
          
        return new_library
    
    def diff_indices(self):
        #returns the different indices and a dict of the list of ds having them
        diff_indices=list()
        indices_repartition=dict()

        #for i in self:
        #    diff_indices.add(self[i].df_data.index)
        #return diff_indices
        
        for i in self :
            if not(check_for_index_in_list(self[i].df_data.index, diff_indices)):
                print('Unique index spotted -', self[i].df_data.index)
                diff_indices.append(copy.deepcopy(self[i].df_data.index))
                indices_repartition[(len(diff_indices)-1)]=[self[i].df_data.index,i]
            else :

                #z=[i for i,x in enumerate(diff_indices) if x.equals(self[i].df_data.index)]
                z=position_of_index_in_list(self[i].df_data.index, diff_indices)
                indices_repartition[z].append(i)

        return diff_indices, indices_repartition   

    def shape_repartitions(self):
        shape_repartition=dict()
        
        for i in self :
            if self[i].df_data.shape in shape_repartition:
                shape_repartition[self[i].df_data.shape].append(self[i].ds_name)
            else :
                shape_repartition[self[i].df_data.shape]=[self[i].ds_name]

        return shape_repartition
    
    def sort_all_indices(self):

        for i in self :
            self[i].df_data.sort_index(inplace=True)
        self.target_ds.sort_index(inplace=True)

    def display_repartition(self):
        for x in self.shape_repartitions():
            print(x,' : ',self.shape_repartitions()[x],'\n')
    
    def __repr__(self):
        number_of_DS=len(list(self.keys()))
        return('DS-Library : '+str(self.library_name)+'\nNumber of DataSets : '+str(number_of_DS))



###########################################################
#Some functions for the building methods

def rewrite_fcm_as_netw(df,jp_csv):
    df.index=jp_csv[2]
    df.index.name='network'
    df.columns=jp_csv[2]
    df.columns.name='network'
    return df

def adapt_fd_to_df_ds_19(df_o,df_ds_19):
    df= df_o.copy() 
    #df= df.transpose().add_suffix('_1').transpose()
    df['ds']=df_ds_19.copy()
    df =df.dropna()
    df =df.drop('ds',axis=1)
    return df

def res_extract(df):
    #Used to extract all the columns wieh 'Age' but not exactly 'Age'
    all_cols=list()
    for key in df.columns:
        if 'Age' in key and key!='Age':
            all_cols.append(key)
    return df[all_cols]

def only_string(df, search='err', must_be=True):
    '''
    Extract all columns with string in it if must_be=True
    Else keep all but these columns
    '''
    all_cols=list()
    for key in df.columns:
        if search in key:
            all_cols.append(key)
    if must_be :
        return df[all_cols]
    else :
        return df.drop(all_cols, axis=1)

def rename_age(df_org, add='_cv_est'):
    df=df_org.copy()
    new_columns=list()
    for key in df.columns:
        new_columns.append(str(key+add))
    df.columns=new_columns
    return df

def transform_index(df_o):
    df = df_o.copy()
    new_index=dict()
    for key in list(df.index):
        new_index[key]=key[:-2]
    df=df.rename(index=new_index)
    df.index = df.index.astype(int)
    return df

def negs_are_nan(x):
    if x < 0:
        return np.nan
    else :
        return x

def transf_col_for_mult(df_o):
    df = df_o.copy()
    new_columns=dict()
    for key in list(df.columns):
        new_columns[key]=key[:-8]
    df=df.rename(columns=new_columns)

    return df

def transf_col_after_mult(df_o):
    df = df_o.copy()
    new_columns=dict()
    for key in list(df.columns):
        new_columns[key]=key+'MultThickAvgAndSurf'
    df=df.rename(columns=new_columns)

    return df

###########################################################
#Building methods for the multiple datasets

def generate_from_jdata(func_folder, our_data_folder):

    d_fcm,d_cod, df_fcm, df_cod_mean, df_fcm_pos_neg= dict(), dict(),dict(),dict(),dict()
    d_fts,df_fts, d_fts_int, df_fts_ints=dict(),dict(),dict(),dict()
    d_fcm_sspn, df_fcm_sspn=dict(),dict()
    df_cod=dict()
    
    sum_pos = lambda x : x[x > 0].sum()
    sum_neg = lambda x : x[x < 0].sum()
        
    for ID in os.listdir(func_folder):
        #Correlation matrices for func. co.
        d_fcm[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_sign_CorrMatrix_z.txt'), sep=" ", header=None)
        df_fcm[ID]=pd.Series(pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_sign_CorrMatrix_z.txt'), sep=" ", header=None).values.flatten(), name=ID)
        sum_all_pos = pd.Series(d_fcm[ID].apply(sum_pos).values.flatten(),name=ID)
        sum_all_neg = pd.Series(d_fcm[ID].apply(sum_neg).values.flatten(),name=ID)
        df_fcm_pos_neg[ID]= sum_all_pos.append(sum_all_neg,ignore_index=True)
        
        #Connectivity matrix for structural co. (tractography)
        d_cod[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None)
        df_cod[ID]=pd.Series( pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None).values.flatten(), name=ID)
        df_cod_mean[ID]=pd.Series( pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None).mean(), name=ID)
        
        #Time series derived values
        d_fts[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_1_GSbptf_meanTS.txt'), sep="\t", header=None)
        d_fts_int[ID]=d_fts[ID].transpose().apply(lambda g:integrate.trapz(g),axis=1)
        df_fts_ints[ID]=d_fts_int[ID].sum()

    df_fts_ints=pd.DataFrame(df_fts_ints, index=['ACTIVITATION_sum_of_ints']).transpose()
    df_fts_ints.index=df_fts_ints.index.astype(int)
    df_fts_ints.to_csv(our_data_folder+'/df_fts_ints.csv')
    
    df_cod_mean=pd.DataFrame(df_cod_mean).transpose()
    df_cod_mean.index=df_cod_mean.index.astype(int)
    df_cod_mean.to_csv(our_data_folder+'/df_cod_mean.csv')
    
    df_fcm=pd.DataFrame.from_dict(df_fcm).transpose()
    df_fcm.index=df_fcm.index.astype(int)
    df_fcm.to_csv(our_data_folder+'/df_fcm.csv')
    
    #Pos. and neg. for all regions
    df_fcm_pos_neg=pd.DataFrame.from_dict(df_fcm_pos_neg).transpose()
    df_fcm_pos_neg.index=df_fcm_pos_neg.index.astype(int)
    df_fcm_pos_neg.to_csv(our_data_folder+'/df_fcm_pos_neg.csv')
    
    #Summing all absolute values of correlation matrices - Marker for organisation ?
    df_fcm_org_calc=pd.DataFrame(df_fcm.abs().transpose().sum())
    df_fcm_org_calc=df_fcm_org_calc.rename(columns={0:'FCM_org'})
    
    #Summing all values (no abs.) of correlation matrices - Marker for organisation ?
    df_fcm_pos_vs_neg_calc=pd.DataFrame(df_fcm.transpose().sum())
    df_fcm_org_calc=df_fcm_org_calc.rename(columns={0:'FCM_org'})
    df_fcm_pos_vs_neg_calc=df_fcm_pos_vs_neg_calc.rename(columns={0:'FCM_pvn'})
    
    #Summing pos values together and neg values together, separated, for eah region (JR)
    df_fcm_pos_and_neg_calc = df_fcm_pos_vs_neg_calc.copy()
    df_fcm_pos_and_neg_calc['FCM_sum_pos'] = df_fcm.transpose().apply(sum_pos)
    df_fcm_pos_and_neg_calc['FCM_sum_neg'] = df_fcm.transpose().apply(sum_neg) 
    df_fcm_pos_and_neg_calc                = df_fcm_pos_and_neg_calc.drop('FCM_pvn', axis=1) 
      
    df_fcm_org_calc.to_csv(our_data_folder+'/df_fcm_org_calc.csv')
    df_fcm_pos_vs_neg_calc.to_csv(our_data_folder+'/df_fcm_pos_vs_neg_calc.csv')
    df_fcm_pos_and_neg_calc.to_csv(our_data_folder+'/df_fcm_pos_and_neg_calc.csv')
    
    df_cod=pd.DataFrame.from_dict(df_cod).transpose()
    df_cod.index = df_cod.index.astype(int)
    df_cod.to_csv(our_data_folder+'/df_cod.csv')
    
    print('iahuuu')

def generate_from_jdata_optimized(func_folder, our_data_folder):

    d_fcm,d_cod, df_fcm = dict(), dict(),dict()
    df_cod=dict()
    
    for ID in os.listdir(func_folder):
        #Correlation matrices for func. co.
        d_fcm[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_sign_CorrMatrix_z.txt'), sep=" ", header=None)
        df_fcm[ID]=pd.Series(pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_sign_CorrMatrix_z.txt'), sep=" ", header=None).values.flatten(), name=ID)
        
        #Connectivity matrix for structural co. (tractography)
        d_cod[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None)
        df_cod[ID]=pd.Series( pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None).values.flatten(), name=ID)
    
    #Functionnal data
    df_fcm=pd.DataFrame.from_dict(df_fcm).transpose()
    df_fcm.index=df_fcm.index.astype(int)
    fcm_regions_number = int(np.sqrt(len(df_fcm.columns)))
    
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    columns_to_keep, flattened =list(),list()
    for i in range(fcm_regions_number):
      columns_to_keep.append(list(range(fcm_regions_number*i,i+fcm_regions_number*i)))
    
    #flattened = [str(x) for x in flatten(columns_to_keep)]
    flattened = [int(x) for x in flatten(columns_to_keep)]
    df_fcm=df_fcm[flattened]
    
    df_fcm.to_csv(our_data_folder+'/df_fcm_optimized.csv')
    
    #Connectivity data
    df_cod=pd.DataFrame.from_dict(df_cod).transpose()
    df_cod.index = df_cod.index.astype(int)
    
    cod_regions_number = int(np.sqrt(len(df_cod.columns)))
    
    columns_to_keep, flattened =list(),list()
    for i in range(cod_regions_number):
      columns_to_keep.append(list(range(cod_regions_number*i,i+cod_regions_number*i)))
    
    #flattened = [str(x) for x in flatten(columns_to_keep)]
    flattened = [int(x) for x in flatten(columns_to_keep)]
    
    df_cod=df_cod[flattened]
    
    df_cod.to_csv(our_data_folder+'/df_cod_optimized.csv')
    
    print('iahuuu')
    
def generate_from_jdata_2(func_folder, our_data_folder):

    d_nd,df_nd= dict(), dict()
    
    for ID in os.listdir(func_folder):
        d_nd[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_1_GSbptf_meanTS.txt'), sep="\t", header=None)
        df_nd[ID]=pd.Series(pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_1_GSbptf_meanTS.txt'), sep="\t", header=None).values.flatten(), name=ID)

    df_nd=pd.DataFrame.from_dict(nd).transpose()
    df_nd.index=df_nd.index.astype(int)
    df_nd.to_csv(our_data_folder+'/df_nd.csv')
    
    print('iahuuu')


def generate_from_jdata_for_each_region(func_folder, our_data_folder):
    #Not activated - uncomment to_csv line for activation
    d_fcm,d_cod, df_fcm, df_cod_mean= dict(), dict(),dict(),dict()
    d_fts,df_fts, d_fts_int, df_fts_ints=dict(),dict(),dict(),dict()
    df_cod=dict()
    one_id=0
    
    for ID in os.listdir(func_folder):
        d_fcm[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_sign_CorrMatrix_z.txt'), sep=" ", header=None)
#        df_fcm[ID]=pd.Series(pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_sign_CorrMatrix_z.txt'), sep=" ", header=None).values.flatten(), name=ID)
        
        d_cod[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None)
#        df_cod[ID]=pd.Series( pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None).values.flatten(), name=ID)
#        df_cod_mean[ID]=pd.Series( pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None).mean(), name=ID)
    
        d_fts[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_1_GSbptf_meanTS.txt'), sep="\t", header=None)
        d_fts_int[ID]=d_fts[ID].transpose().apply(lambda g:integrate.trapz(g),axis=1)
        df_fts_ints[ID]=d_fts_int[ID].sum()
        one_id=ID
        
    regions = list(d_fcm[one_id].columns)
    regional_cms=dict()
    
    for region in regions:
        regional_cms[region]= dict()
        for ID in os.listdir(func_folder):
            regional_cms[region][int(ID)]=d_fcm[ID][region]
        regional_cms[region] = pd.DataFrame.from_dict(regional_cms[region]).transpose()
        regional_cms[region].index = regional_cms[region].index.astype(int)
        regional_cms[region].to_csv(our_data_folder+'/'+'df_fcm_region_'+str(region)+'.csv')
    
    regions = list(d_cod[one_id].columns)
    regional_cod=dict()
    
    for region in regions:
        regional_cod[region]= dict()
        for ID in os.listdir(func_folder):
            regional_cod[region][int(ID)]=d_cod[ID][region]
        regional_cod[region] = pd.DataFrame.from_dict(regional_cod[region]).transpose()
        regional_cod[region].index = regional_cod[region].index.astype(int)
        regional_cod[region].to_csv(our_data_folder+'/'+'df_cod_region'+str(region)+'.csv')
        
    return regional_cms, regional_cod

    
    
#    df_fts_ints=pd.DataFrame(df_fts_ints, index=['ACTIVITATION_sum_of_ints']).transpose()
#    df_fts_ints.index=df_fts_ints.index.astype(int)
##    df_fts_ints.to_csv(our_data_folder+'/df_fts_ints.csv')
#    
#    df_cod_mean=pd.DataFrame(df_cod_mean).transpose()
#    df_cod_mean.index=df_cod_mean.index.astype(int)
##    df_cod_mean.to_csv(our_data_folder+'/df_cod_mean.csv')
#    
#    df_fcm=pd.DataFrame.from_dict(df_fcm).transpose()
#    df_fcm.index=df_fcm.index.astype(int)
##    df_fcm.to_csv(our_data_folder+'/df_fcm.csv')
#    
#    #Summing all absolute values of correlation matrices - Marker for organisation ?
#    df_fcm_org_calc=pd.DataFrame(df_fcm.abs().transpose().sum())
#    df_fcm_org_calc=df_fcm_org_calc.rename(columns={0:'FCM_org'})
#    
#    df_fcm_pos_vs_neg_calc=pd.DataFrame(df_fcm.transpose().sum())
#    df_fcm_org_calc=df_fcm_org_calc.rename(columns={0:'FCM_org'})
#    df_fcm_pos_vs_neg_calc=df_fcm_pos_vs_neg_calc.rename(columns={0:'FCM_pvn'})
#    
#    df_fcm_org_calc.to_csv(our_data_folder+'/df_fcm_org_calc.csv')
##    df_fcm_pos_vs_neg_calc.to_csv(our_data_folder+'/df_fcm_pos_vs_neg_calc.csv')
#    
#    df_cod=pd.DataFrame.from_dict(df_cod).transpose()
#    df_cod.index = df_cod.index.astype(int)
#    df_cod.to_csv(our_data_folder+'/df_cod.csv')
    
   
###########################################################
#Old nasty way of getting the data

def build_function_for_control_df_ct(our_data_folder,private_data_folder='./../../data'):
    get_cwd = os.getcwd()
    os.chdir(os.path.dirname(private_data_folder))
    name='Age'
#    classification=True
#    scorer=lambda x:x
#    greater_is_better=True
    gender = None    
    s_filter=ds_2CAT
    drop=[]
    target='Age'
    data_choice = 'ct'
#    config_dict=None
#    pop = 250
#    gen = 20
    th = 1
#    save=True
#    save_d=True
#    c_display=0
    norm=False
    clean=False
    dynamic_drop = False
    factor = 3
    ban_thresh = 10
    undersample = None
    age =  None
    rand = 23        
    
    control_df_ct = data_preprocessing(name=name, s_filter=s_filter, 
                                drop=drop, data_choice = data_choice, 
                                normalization=norm ,clean= clean, th=th,
                                factor = factor, dynamic_drop=dynamic_drop,
                                ban_thresh = ban_thresh, gender=gender, age=age, 
                                undersample =undersample, rand=rand, target=target, 
                                )
    control_df_ct=control_df_ct.sort_index()
    control_df_ct.to_csv(our_data_folder+'/control_df_ct.csv')
    os.chdir(get_cwd)
            
def build_function_for_control_df_cv(our_data_folder,private_data_folder='./../../data'):
    name='Age'
    get_cwd = os.getcwd()
    os.chdir(os.path.dirname(private_data_folder))
#    classification=True
#    scorer=lambda x:x
#    greater_is_better=True
    gender = None    
    s_filter=ds_2CAT
    drop=[]
    target='Age'
    data_choice = 'cv'
#    config_dict=None
#    pop = 250
#    gen = 20
    th = 1
#    save=True
#    save_d=True
#    c_display=0
    norm=False
    clean=False
    dynamic_drop = False
    factor = 3
    ban_thresh = 10
    undersample = None
    age =  None
    rand = 23        
    
    control_df_cv = data_preprocessing(name=name, s_filter=s_filter, 
                                drop=drop, data_choice = data_choice, 
                                normalization=norm ,clean= clean, th=th,
                                factor = factor, dynamic_drop=dynamic_drop,
                                ban_thresh = ban_thresh, gender=gender, age=age, 
                                undersample =undersample, rand=rand, target=target, 
                                )
    control_df_cv=control_df_cv.sort_index()
    control_df_cv.to_csv(our_data_folder+'/control_df_cv.csv')
    os.chdir(get_cwd)

###########################################################

def build_method_for_na(private_data_folder, our_data_folder):
    
    #get_cwd = os.getcwd()
    #os.chdir(os.path.dirname(private_data_folder))
    
    df_gv_na=transform_index(pd.read_csv(private_data_folder+'/'+'1000BRAINS_NAD_GrayVol.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_GrayVol',axis=1))
    df_mc_na=transform_index(pd.read_csv(private_data_folder+'/'+'1000BRAINS_NAD_MeanCurv.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_MeanCurv',axis=1))
    df_nv_na=transform_index(pd.read_csv(private_data_folder+'/'+'1000BRAINS_NAD_NumVert.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_NumVert',axis=1))
    df_sa_na=transform_index(pd.read_csv(private_data_folder+'/'+'1000BRAINS_NAD_SurfArea.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_SurfArea',axis=1))
    df_ta_na=transform_index(pd.read_csv(private_data_folder+'/'+'1000BRAINS_NAD_ThickAvg.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_ThickAvg',axis=1))
    df_ts_na=transform_index(pd.read_csv(private_data_folder+'/'+'1000BRAINS_NAD_ThickStd.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_ThickStd',axis=1))
    df_fi_na=transform_index(pd.read_csv(private_data_folder+'/'+'1000BRAINS_NAD_FoldInd.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_FoldInd',axis=1))
    df_gc_na=transform_index(pd.read_csv(private_data_folder+'/'+'1000BRAINS_NAD_GausCurv.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_GausCurv',axis=1))
    df_ci_na=transform_index(pd.read_csv(private_data_folder+'/'+'1000BRAINS_NAD_CurvInd.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_CurvInd',axis=1))
    
    df_mul_ta_sa_na=transf_col_after_mult(transf_col_for_mult(df_ta_na) * transf_col_for_mult(df_sa_na))
    
    df_mul_ta_sa_na.to_csv(our_data_folder+'/df_mul_ta_sa_na.csv')
    
    df_gv_na.to_csv(our_data_folder+'/df_gv_na.csv')
    df_mc_na.to_csv(our_data_folder+'/df_mc_na.csv')
    df_nv_na.to_csv(our_data_folder+'/df_nv_na.csv')
    df_sa_na.to_csv(our_data_folder+'/df_sa_na.csv')
    df_ta_na.to_csv(our_data_folder+'/df_ta_na.csv')
    df_ts_na.to_csv(our_data_folder+'/df_ts_na.csv')
    df_fi_na.to_csv(our_data_folder+'/df_fi_na.csv')
    df_gc_na.to_csv(our_data_folder+'/df_gc_na.csv')
    df_ci_na.to_csv(our_data_folder+'/df_ci_na.csv')
    
    #os.chdir(get_cwd)
    
def build_method_for_gen_inf(private_data_folder, our_data_folder):
    
    df_gen=pd.read_csv\
        (private_data_folder+'/FB1000G_BDI_2019-04-01_09-33-35.csv', index_col='SubjectID', sep='\t')
    
    df_gen.index.names=['ID']
    df_gen = df_gen[['Gender','Age', 'SCOREBDI']]
    df_gen['Gender']    =    df_gen['Gender'].apply(gender_to_b)
    df_gen['SCOREBDI']  =    df_gen['SCOREBDI'].apply(negs_are_nan)
    
    df_gen.to_csv(our_data_folder+'/df_gen.csv')
    print('General data successfully generated!')

def build_method_for_bdi_scores(our_data_folder):
    
    df_gen=pd.read_csv\
        (our_data_folder+'/df_gen.csv')
    df_ds_10=   df_gen['SCOREBDI'].apply(lambda x: ds_2CAT_b(x,10))
    df_ds_19=   df_gen['SCOREBDI'].apply(lambda x: ds_2CAT_b(x,19))
    df_ds_10_cs=df_gen['SCOREBDI'].apply(lambda x: ds_cs_b(x,10))
    df_ds_19_cs=df_gen['SCOREBDI'].apply(lambda x: ds_cs_b(x,19))
    
    df_ds_10.columns = pd.Index(['ds'])
    df_ds_19.columns = pd.Index(['ds'])
    df_ds_10_cs.columns = pd.Index(['ds'])
    df_ds_19_cs.columns = pd.Index(['ds'])
    
    df_ds_10=df_ds_10.sort_index()
    df_ds_19=df_ds_19.sort_index()
    df_ds_10_cs=df_ds_10_cs.sort_index()
    df_ds_19_cs=df_ds_19_cs.sort_index()
    
    df_ds_10.to_csv(our_data_folder+'/df_ds_10.csv')
    df_ds_19.to_csv(our_data_folder+'/df_ds_19.csv')
    df_ds_10_cs.to_csv(our_data_folder+'/df_ds_10_cs.csv')
    df_ds_19_cs.to_csv(our_data_folder+'/df_ds_19_cs.csv')
    print('BDI data successfully generated!')


###########################################################

def building_general_attributes(func_folder=func_folder, our_data_folder=our_data_folder, private_data_folder=private_data_folder):
    directory = 'add_freesurfer_direct'
    alldirs     = [i for i in os.listdir(directory) if '_1' in i]
    subfolders  = [i for i in alldirs if i.split('_1')[1] == '']
    
    file_where_extract = 'aseg.stats'
    extracted_features = dict()
    
    for ID in subfolders :
      folder= directory + '/' + ID + '/' + 'stats' + '/' + file_where_extract 
      infile = open(folder,'r')
      file_as_str = infile.read()
      infile.close()
      extracted_features[int(ID[:-2])] = dict()
      extracted_features[int(ID[:-2])]['eTIV']  =   (float(file_as_str.split('\n')[34].split(',')[-2]))
      extracted_features[int(ID[:-2])]['TotalGreyMatter']  =   (float(file_as_str.split('\n')[24].split(',')[-2]))
      extracted_features[int(ID[:-2])]['TotalWhiteMatter']  =   (float(file_as_str.split('\n')[22].split(',')[-2]))
      extracted_features[int(ID[:-2])]['BrainSegNotVent ']  =   (float(file_as_str.split('\n')[13].split(',')[-2]))
      extracted_features[int(ID[:-2])]['BrainSegNotVent ']  =   (float(file_as_str.split('\n')[13].split(',')[-2]))
      
      #Added 13.03.2020. Adds measures of rh and lh TGM and TWM
      extracted_features[int(ID[:-2])]['LH_TWM']  =   (float(file_as_str.split('\n')[20].split(',')[-2]))
      extracted_features[int(ID[:-2])]['LH_TGM']  =   (float(file_as_str.split('\n')[17].split(',')[-2]))
      extracted_features[int(ID[:-2])]['RH_TWM']  =   (float(file_as_str.split('\n')[21].split(',')[-2]))
      extracted_features[int(ID[:-2])]['RH_TGM']  =   (float(file_as_str.split('\n')[18].split(',')[-2]))
      
      
    df = pd.DataFrame.from_dict(extracted_features, orient='index')
    df.to_csv(our_data_folder+'/general_anat_feats.csv')

###########################################################

def building_from_CAT(func_folder=func_folder, our_data_folder=our_data_folder, private_data_folder=private_data_folder):
    directory = 'add_cat_directory'
    subdir = '/CAT/TIV_'
    suffix = '.txt'
    
    alldirs     = [i for i in os.listdir(directory) if 'sub-' in i]
    #all_tiv_files  = [directory+i+subdir+i+suffix for i in alldirs]
    
    errors = []
    extracted_features = dict()
    
    for ID in alldirs :
      print('CAT files tracked for : ', ID)
      try :
        folder= directory+ID+subdir+ID+suffix 
        infile = open(folder,'r')
        file_as_str = infile.read()
        infile.close()
        extracted_features[int(ID[4:])] = dict()
        extracted_features[int(ID[4:])]['eTIV_spm']  =   float(file_as_str.split('\t')[0])
        extracted_features[int(ID[4:])]['TGM_spm']  =   float(file_as_str.split('\t')[1])
        extracted_features[int(ID[4:])]['TWM_spm']  =   float(file_as_str.split('\t')[2])
        extracted_features[int(ID[4:])]['CSF_spm']  =   float(file_as_str.split('\t')[3])
      except OSError as e:
        print(e)
        print('Could not find TIV for ',ID)
        errors.append(ID)
        continue
    errors = '\n'.join(errors)
    with open(our_data_folder+'/cat_errors.txt','w') as written_file:
      written_file.write(errors)
    df = pd.DataFrame.from_dict(extracted_features, orient='index')
    df.to_csv(our_data_folder+'/cat_derivated_feats.csv')

###########################################################

def run_builders(func_folder=func_folder, our_data_folder=our_data_folder, private_data_folder=private_data_folder):
    generate_from_jdata(func_folder, our_data_folder)
    #build_function_for_control_df_ct(our_data_folder)
    #build_function_for_control_df_cv(our_data_folder)
    
    build_method_for_na(private_data_folder, our_data_folder)
    build_method_for_gen_inf(private_data_folder, our_data_folder)
    build_method_for_bdi_scores(our_data_folder)

###########################################################



def build_method_for_previous_datasets(our_data_folder):

    df_ct = pd.read_csv('regression_ct_battery_with_err.csv',index_col ='ID')
    df_cv = pd.read_csv('regression_cv_battery_with_err.csv',index_col ='ID')
    df_na = transform_index(pd.read_csv('1000BRAINS_nad3.csv', index_col='ID').drop(['Gender','Age','SCOREBDI'], axis=1))
    df_na['Gender'] = df_na['Gender'].apply(lambda x: 0 if x =='Male' else 1)
    df_na.index = df_na.index.astype(int)
    
    df_cv=only_string(df_cv, 'Age',must_be=False)
    df_ct=only_string(df_ct, 'Age',must_be=False)
    
    df_cv=only_string(df_cv,'volume',must_be=True)
    df_ct=only_string(df_ct,'thickness',must_be=True)
    df_gen_struct=pd.DataFrame()
    df_gen_struct['glob_vol']=df_cv.sum(axis=1)
    df_gen_struct['mean_thick']=df_ct.mean(axis=1)
    
    df_cvr = rename_age(res_extract(df_cv))
    df_ctr = rename_age(res_extract(df_ct),add='_ct_est')
    
    df_cvr_err = rename_age(only_string(res_extract(df_cv)))
    df_ctr_err = rename_age(only_string(res_extract(df_ct)),add='_ct_est')
    
    df_cvr_tr = rename_age(only_string(res_extract(df_cv), must_be='False'))
    df_ctr_tr = rename_age(only_string(res_extract(df_ct), must_be='False'), \
                           add='_ct_est')
    
    df_both_err=pd.concat([df_cvr_err,df_ctr_err],axis=1)
    df_both_tr=pd.concat([df_cvr_tr,df_ctr_tr],axis=1)
    
    df_ct=df_ct.sort_index()
    df_cv=df_cv.sort_index()
    df_cvr=df_cvr.sort_index()
    df_ctr=df_ctr.sort_index()
    df_cvr_err=df_cvr_err.sort_index()
    df_ctr_err=df_ctr_err.sort_index()
    df_both_err=df_both_err.sort_index()
    
    df_all_together = pd.concat([df_cv,df_ct,df_cvr,df_ctr,df_gen],axis=1,sort=True)
    
    missing_id, valid_id_list=list(),list(df_ct.index)
    
    
    
    for element in list(control_df_ct.index):
        if element not in valid_id_list:
            missing_id.append(element)
    
    #df_na['ds']=df_ds_19
    #df_na=df_na.dropna()
    #df_na=df_na.drop('ds',axis=1)
    
    new_index= list()
    #df_cod_mean = df_cod_mean.transpose().add_suffix('_1').transpose()
    #df_cod_mean['ds']=df_ds_19
    #df_cod_mean=df_cod_mean.dropna()
    #df_cod_mean=df_cod_mean.drop('ds',axis=1)
    
    control_df_ct=control_df_ct.drop(missing_id)
    control_df_cv=control_df_cv.drop(missing_id)
    
    #df_cod_mean_a = adapt_fd_to_df_ds_19(df_cod_mean,df_ds_19)
    #df_cod_hv_a = adapt_fd_to_df_ds_19(df_cod_hv,df_ds_19)
    df_fcm_a = adapt_fd_to_df_ds_19(df_fcm,df_ds_19)
    
    assert (control_df_cv.index == df_ct.index).all()
    assert (control_df_ct.index == df_ct.index).all()
    
    control_df_ct[df_ct.columns] == df_ct
    control_df_cv[df_cv.columns] == df_cv
    
    assert (df_cv.index == df_ct.index).all()
    #assert (df_gen.index == df_ct.index).all()
    assert (df_cvr.index == df_ct.index).all()
    assert (df_ctr.index == df_ct.index).all()
    #assert (df_ds_10.index == df_ct.index).all()
    #assert (df_ds_19.index == df_ct.index).all()
    #assert (df_all_together.index == df_ct.index).all()
    assert (df_cvr_err.index == df_ct.index).all()
    assert (df_ctr_err.index == df_ct.index).all()
    assert (df_both_err.index == df_ct.index).all()
    #assert (df_na.index == df_ct.index).all()


