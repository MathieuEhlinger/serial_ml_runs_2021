# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:08:35 2019

@author: Mathieu

Just loading a big bunch of datasets in memory

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

from datai.preprocessing import data_preprocessing

from datai.ds_filters import ds_4CAT, ds_cs_b, ds_2CAT,ds_2CAT_b, gender_to_b

j_parcellation='add_j_parcellation_path'
func_folder = 'add_func_data_folder'
general_inf_file = 'add_general_info_file'

jp_csv=pd.read_csv(j_parcellation, header=None)

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

###########################################################

#os.chdir(func_folder)

d_fcm,d_cod, df_fcm, df_cod_mean= dict(), dict(),dict(),dict()
d_fts,df_fts, d_fts_int, df_fts_ints=dict(),dict(),dict(),dict()
df_cod=dict()

for ID in os.listdir(func_folder):
    d_fcm[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_sign_CorrMatrix_z.txt'), sep=" ", header=None)
    df_fcm[ID]=pd.Series(pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_sign_CorrMatrix_z.txt'), sep=" ", header=None).values.flatten(), name=ID)
    
    d_cod[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None)
    df_cod[ID]=pd.Series( pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None).values.flatten(), name=ID)
    df_cod_mean[ID]=pd.Series( pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None).mean(), name=ID)

    d_fts[ID]=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_1_GSbptf_meanTS.txt'), sep="\t", header=None)
    d_fts_int[ID]=d_fts[ID].transpose().apply(lambda g:integrate.trapz(g),axis=1)
    df_fts_ints[ID]=d_fts_int[ID].sum()

df_fts_ints=pd.DataFrame(df_fts_ints, index=['ACTIVITATION_sum_of_ints']).transpose()
df_fts_ints.index=df_fts_ints.index.astype(int)

df_cod_mean=pd.DataFrame(df_cod_mean).transpose()
df_cod_mean.index=df_cod_mean.index.astype(int)

df_fcm=pd.DataFrame.from_dict(df_fcm).transpose()
df_fcm.index=df_fcm.index.astype(int)

#Summing all absolute values of correlation matrices - Marker for organisation ?
df_fcmorg_calc=pd.DataFrame(df_fcm.abs().transpose().sum())
df_fcmorg_calc=df_fcmorg_calc.rename(columns={0:'FCM_org'})

df_fcm_pos_vs_neg_calc=pd.DataFrame(df_fcm.transpose().sum())
df_fcmorg_calc=df_fcmorg_calc.rename(columns={0:'FCM_org'})
df_fcm_pos_vs_neg_calc=df_fcm_pos_vs_neg_calc.rename(columns={0:'FCM_pvn'})

df_cod=pd.DataFrame.from_dict(df_cod).transpose()
df_cod.index = df_cod.index.astype(int)
print('iahuuu')
###########################################################
#Old nasty way of getting the data

name='Age'
classification=True
scorer=lambda x:x
greater_is_better=True
gender = None    
s_filter=ds_2CAT
drop=[]
target='Age'
data_choice = 'ct'
config_dict=None
pop = 250
gen = 20
th = 1
save=True
save_d=True
c_display=0
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

name='Age'
classification=True
scorer=lambda x:x
greater_is_better=True
gender = None    
s_filter=ds_2CAT
drop=[]
target='Age'
data_choice = 'cv'
config_dict=None
pop = 250
gen = 20
th = 1
save=True
save_d=True
c_display=0
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
control_df_ct=control_df_ct.sort_index()
control_df_cv=control_df_cv.sort_index()

###########################################################



def transform_index(df_o):
    df = df_o.copy()
    new_index=dict()
    for key in list(df.index):
        new_index[key]=key[:-2]
    df=df.rename(index=new_index)
    df.index = df.index.astype(int)
    return df

def negs_are_nan(x,column='SCOREBDI'):
    if x < 0:
        return np.nan
    else :
        return x


    

get_cwd = os.getcwd()
os.chdir(os.path.dirname(__file__))
df_ct = pd.read_csv('regression_ct_battery_with_err.csv',index_col ='ID')
df_cv = pd.read_csv('regression_cv_battery_with_err.csv',index_col ='ID')
df_na = transform_index(pd.read_csv('1000BRAINS_nad3.csv', index_col='ID').drop(['Gender','Age','SCOREBDI'], axis=1))
#df_na['Gender'] = df_na['Gender'].apply(lambda x: 0 if x =='Male' else 1)

df_gv_na=transform_index(pd.read_csv('1000BRAINS_NAD_GrayVol.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_GrayVol',axis=1))
df_mc_na=transform_index(pd.read_csv('1000BRAINS_NAD_MeanCurv.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_MeanCurv',axis=1))
df_nv_na=transform_index(pd.read_csv('1000BRAINS_NAD_NumVert.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_NumVert',axis=1))
df_sa_na=transform_index(pd.read_csv('1000BRAINS_NAD_SurfArea.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_SurfArea',axis=1))
df_ta_na=transform_index(pd.read_csv('1000BRAINS_NAD_ThickAvg.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_ThickAvg',axis=1))
df_ts_na=transform_index(pd.read_csv('1000BRAINS_NAD_ThickStd.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_ThickStd',axis=1))
df_fi_na=transform_index(pd.read_csv('1000BRAINS_NAD_FoldInd.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_FoldInd',axis=1))
df_gc_na=transform_index(pd.read_csv('1000BRAINS_NAD_GausCurv.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_GausCurv',axis=1))
df_ci_na=transform_index(pd.read_csv('1000BRAINS_NAD_CurvInd.csv', index_col='Unnamed: 0', sep=',').drop('lh.CT_*_CurvInd',axis=1))
df_na.index = df_na.index.astype(int)
df_gen=pd.read_csv(general_inf_file, index_col='SubjectID', sep='\t')
os.chdir(get_cwd)

df_gvsum_na=pd.DataFrame(df_gv_na.transpose().sum(), columns=['GV_total'])

df_tamean_na=pd.DataFrame([df_ta_na.transpose().mean(),df_ta_na.transpose().median()], index=['CTA_mean','CTA_median']).transpose()

df_gen = df_gen[['Gender','Age', 'SCOREBDI']]
        #final_df = final_df.divide(i).sort_values(by='balanced accuracy',ascending=False)
df_gen.index.names=['ID']
df_gen['Gender']=df_gen['Gender'].apply(gender_to_b)
df_gen['SCOREBDI'] = df_gen['SCOREBDI'].apply(negs_are_nan)

df_cvr = rename_age(res_extract(df_cv))
df_ctr = rename_age(res_extract(df_ct),add='_ct_est')

df_cvr_err = rename_age(only_string(res_extract(df_cv)))
df_ctr_err = rename_age(only_string(res_extract(df_ct)),add='_ct_est')

df_cvr_tr = rename_age(only_string(res_extract(df_cv), must_be='False'))
df_ctr_tr = rename_age(only_string(res_extract(df_ct), must_be='False'), \
                       add='_ct_est')

df_both_err=pd.concat([df_cvr_err,df_ctr_err],axis=1)
df_both_tr=pd.concat([df_cvr_tr,df_ctr_tr],axis=1)

#df_ds_10, df_ds_19 = pd.DataFrame(), pd.DataFrame()
df_bdi = df_gen[['SCOREBDI']]
df_all_anat= pd.concat([df_ci_na,df_fi_na,df_gc_na,df_gv_na,df_mc_na,df_nv_na,df_sa_na,df_ta_na,df_ts_na], axis=1)

df_ds_10=df_gen['SCOREBDI'].apply(lambda x: ds_2CAT_b(x,10))
df_ds_19=df_gen['SCOREBDI'].apply(lambda x: ds_2CAT_b(x,19))
df_ds_10_cs=df_gen['SCOREBDI'].apply(lambda x: ds_cs_b(x,10))
df_ds_19_cs=df_gen['SCOREBDI'].apply(lambda x: ds_cs_b(x,19))

df_ds_10.columns = pd.Index(['ds'])
df_ds_19.columns = pd.Index(['ds'])
df_ds_10_cs.columns = pd.Index(['ds'])
df_ds_10_cs.columns = pd.Index(['ds'])

df_cv=only_string(df_cv, 'Age',must_be=False)
df_ct=only_string(df_ct, 'Age',must_be=False)
#
df_cv=only_string(df_cv,'volume',must_be=True)
df_ct=only_string(df_ct,'thickness',must_be=True)
df_gen_struct=pd.DataFrame()
df_gen_struct['glob_vol']=df_cv.sum(axis=1)
df_gen_struct['mean_thick']=df_ct.mean(axis=1)

df_gen = df_gen.sort_index()
df_ct=df_ct.sort_index()
df_cv=df_cv.sort_index()
df_cvr=df_cvr.sort_index()
df_ctr=df_ctr.sort_index()
df_ds_10=df_ds_10.sort_index()
df_ds_19=df_ds_19.sort_index()
df_cvr_err=df_cvr_err.sort_index()
df_ctr_err=df_ctr_err.sort_index()
df_both_err=df_both_err.sort_index()

df_all_together = pd.concat([df_cv,df_ct,df_cvr,df_ctr,df_gen],axis=1,sort=True)

missing_id, valid_id_list=list(),list(df_ct.index)

df_test = pd.concat([df_tamean_na,df_gvsum_na,df_fcmorg_calc,df_fcm_pos_vs_neg_calc,df_fts_ints,df_gen],axis=1)

for element in list(control_df_ct.index):
    if element not in valid_id_list:
        missing_id.append(element)

#df_na['ds']=df_ds_19
#df_na=df_na.dropna()
#df_na=df_na.drop('ds',axis=1)

new_index= list()


control_df_ct=control_df_ct.drop(missing_id)
control_df_cv=control_df_cv.drop(missing_id)


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

os.chdir(get_cwd)
