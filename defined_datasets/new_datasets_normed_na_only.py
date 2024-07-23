# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:08:35 2019

@author: Mathieu

Just loading a big bunch of datasets in memory

"""

from collections import OrderedDict
import pandas as pd
import os
from datai.ds_building_functions import DSLibrary, DataSet, our_data_folder, func_folder
import copy
from sklearn.preprocessing import normalize

normalizing_func = lambda x:normalize(x,norm='l1',axis=1)

default_ds_lib=         DSLibrary('Default DS library')
spec_ds_lib=            DSLibrary('Objective DS library')
spec_old_ds_lib=        DSLibrary('Old objective DS library')
derived_ds_lib=         DSLibrary('Derived DS Library')
informative_ds_lib=     DSLibrary('General information DS Library')
test_ds_lib=            DSLibrary('Test DS library')
test_ds_lib_2=          DSLibrary('Test DS library')

#General data, like Gender, SCOREBDI, Age.
df_gen=                 DataSet(our_data_folder, 'df_gen', informative_ds_lib).df_data
#New line was added on the 14.11.2019 and adds general Volume information
df_gen = pd.concat([df_gen, pd.DataFrame.from_csv(our_data_folder+'/general_anat_feats.csv')],axis=1)
global_target=          'Age'

default_ds_lib.inf_ds      =     copy.deepcopy(df_gen)       
spec_ds_lib.inf_ds         =     copy.deepcopy(df_gen)
spec_old_ds_lib.inf_ds     =     copy.deepcopy(df_gen)
derived_ds_lib.inf_ds      =     copy.deepcopy(df_gen) 
informative_ds_lib.inf_ds  =     copy.deepcopy(df_gen) 
test_ds_lib.inf_ds         =     copy.deepcopy(df_gen)
test_ds_lib_2.inf_ds         =     copy.deepcopy(df_gen)

default_ds_lib.target_ds=         copy.deepcopy(df_gen[[global_target]])       
spec_ds_lib.target_ds=            copy.deepcopy(df_gen[[global_target]]) 
spec_old_ds_lib.target_ds=        copy.deepcopy(df_gen[[global_target]]) 
derived_ds_lib.target_ds=         copy.deepcopy(df_gen[[global_target]]) 
informative_ds_lib.target_ds=     copy.deepcopy(df_gen[[global_target]])
test_ds_lib.target_ds=            copy.deepcopy(df_gen[[global_target]])
test_ds_lib_2.target_ds=            copy.deepcopy(df_gen[[global_target]])

#A control dataset. Somewhat outdated.
#control_df_ct=          DataSet(our_data_folder, 'control_df_ct', dataset_lib=spec_old_ds_lib).df_data
#control_df_cv=          DataSet(our_data_folder, 'control_df_cv', dataset_lib=spec_old_ds_lib).df_data


df_bdi =                DataSet(df_gen[['SCOREBDI']], 'df_bdi', description='SCOREBDI as ints', dataset_lib=    informative_ds_lib).df_data
df_ds_10=               DataSet(our_data_folder, 'df_ds_10', description='1 if BDI>10', dataset_lib=               informative_ds_lib).df_data
df_ds_19=               DataSet(our_data_folder, 'df_ds_19', dataset_lib=               informative_ds_lib).df_data
df_ds_10_cs=            DataSet(our_data_folder, 'df_ds_10_cs', dataset_lib=            informative_ds_lib).df_data
df_ds_19_cs=            DataSet(our_data_folder, 'df_ds_19_cs', dataset_lib=            informative_ds_lib).df_data

#Connectome and functionnal data.
'''
df_cod_mean=            DataSet(our_data_folder, 'df_cod_mean', dataset_lib=            spec_ds_lib,description= 'Cod - Tracts per region').df_data
df_fts_ints=            DataSet(our_data_folder, 'df_fts_ints', dataset_lib=            spec_ds_lib,description= 'FTS - Summed activations').df_data
df_fcm=                 DataSet(our_data_folder, 'df_fcm', dataset_lib=                 spec_ds_lib,description= 'FCM - Complete matrix').df_data
df_fcm_org_calc=         DataSet(our_data_folder, 'df_fcm_org_calc', dataset_lib=       spec_ds_lib,description= 'FCM - Sum of absolute values for each region of fcm').df_data
df_fcm_pos_vs_neg_calc= DataSet(our_data_folder, 'df_fcm_pos_vs_neg_calc', dataset_lib= spec_ds_lib,description= 'FCM - Sum of all values for each region (+ vs -)').df_data
df_cod=                 DataSet(our_data_folder, 'df_cod', dataset_lib=                 spec_ds_lib,description= 'Cod - Complete matrix').df_data
'''

#df_fts_ints             -> Integrated time series in one value
#df_fcmorg_calc          -> Dunno. Looks pretty boring. Was it the abs value of corr ?
#df_fcm_pos_vs_neg_calc  -> Sum of neg. and pos. in corr. matrix.

#Anatomical data.
gen_anat_ds=pd.DataFrame(index=df_gen.index)
#normalize(new_genme.library['df_gv_na'].df_data,norm='l1',axis=1)[0].sum()

df_gv_na=               DataSet(our_data_folder, 'df_gv_na', dataset_lib=               spec_ds_lib, description='Gray Matter Volume').df_data
gen_anat_ds['sum_gv']=spec_ds_lib['df_gv_na'].df_data.sum(axis=1)
spec_ds_lib['df_gv_na'].df_data = pd.DataFrame(normalizing_func(spec_ds_lib['df_gv_na'].df_data), columns=spec_ds_lib['df_gv_na'].df_data.columns ,\
index=spec_ds_lib['df_gv_na'].df_data.index)

df_mc_na=               DataSet(our_data_folder, 'df_mc_na', dataset_lib=               spec_ds_lib, description='MeanCurv').df_data
spec_ds_lib['df_mc_na'].df_data = pd.DataFrame(normalizing_func(spec_ds_lib['df_mc_na'].df_data), columns=spec_ds_lib['df_mc_na'].df_data.columns ,\
index=spec_ds_lib['df_mc_na'].df_data.index )

df_nv_na=               DataSet(our_data_folder, 'df_nv_na', dataset_lib=               spec_ds_lib, description='NumVert').df_data
gen_anat_ds['sum_nv']=spec_ds_lib['df_nv_na'].df_data.sum(axis=1)
spec_ds_lib['df_nv_na'].df_data = pd.DataFrame(normalizing_func(spec_ds_lib['df_nv_na'].df_data), columns=spec_ds_lib['df_nv_na'].df_data.columns ,\
index=spec_ds_lib['df_nv_na'].df_data.index )

df_sa_na=               DataSet(our_data_folder, 'df_sa_na', dataset_lib=               spec_ds_lib, description='SurfArea').df_data
gen_anat_ds['sum_sa']=spec_ds_lib['df_sa_na'].df_data.sum(axis=1)
spec_ds_lib['df_sa_na'].df_data = pd.DataFrame(normalizing_func(spec_ds_lib['df_sa_na'].df_data), columns=spec_ds_lib['df_sa_na'].df_data.columns ,\
index=spec_ds_lib['df_sa_na'].df_data.index )

df_ta_na=               DataSet(our_data_folder, 'df_ta_na', dataset_lib=               spec_ds_lib, description='ThickAvg').df_data
gen_anat_ds['sum_ta']=spec_ds_lib['df_ta_na'].df_data.sum(axis=1)
spec_ds_lib['df_ta_na'].df_data = pd.DataFrame(normalizing_func(spec_ds_lib['df_ta_na'].df_data), columns=spec_ds_lib['df_ta_na'].df_data.columns ,\
index=spec_ds_lib['df_ta_na'].df_data.index )

df_ts_na=               DataSet(our_data_folder, 'df_ts_na', dataset_lib=               spec_ds_lib, description='ThickStd').df_data
gen_anat_ds['sum_ts']=spec_ds_lib['df_ts_na'].df_data.sum(axis=1)
spec_ds_lib['df_ts_na'].df_data = pd.DataFrame(normalizing_func(spec_ds_lib['df_ts_na'].df_data), columns=spec_ds_lib['df_ts_na'].df_data.columns ,\
index=spec_ds_lib['df_ts_na'].df_data.index )

df_fi_na=               DataSet(our_data_folder, 'df_fi_na', dataset_lib=               spec_ds_lib, description='FoldInd').df_data
gen_anat_ds['sum_fi']=spec_ds_lib['df_fi_na'].df_data.sum(axis=1)
spec_ds_lib['df_fi_na'].df_data = pd.DataFrame(normalizing_func(spec_ds_lib['df_fi_na'].df_data), columns=spec_ds_lib['df_fi_na'].df_data.columns ,\
index=spec_ds_lib['df_fi_na'].df_data.index )

df_gc_na=               DataSet(our_data_folder, 'df_gc_na', dataset_lib=               spec_ds_lib, description='GausCurv').df_data
gen_anat_ds['sum_gc']=spec_ds_lib['df_gc_na'].df_data.sum(axis=1)
spec_ds_lib['df_gc_na'].df_data = pd.DataFrame(normalizing_func(spec_ds_lib['df_gc_na'].df_data), columns=spec_ds_lib['df_gc_na'].df_data.columns ,\
index=spec_ds_lib['df_gc_na'].df_data.index )

df_ci_na=               DataSet(our_data_folder, 'df_ci_na', dataset_lib=               spec_ds_lib, description='CurvInd').df_data
gen_anat_ds['sum_ci']=spec_ds_lib['df_ci_na'].df_data.sum(axis=1)
spec_ds_lib['df_ci_na'].df_data = pd.DataFrame(normalizing_func(spec_ds_lib['df_ci_na'].df_data), columns=spec_ds_lib['df_ci_na'].df_data.columns ,\
index=spec_ds_lib['df_ci_na'].df_data.index )

df_na =                 DataSet(our_data_folder, 'df_na'   , dataset_lib=               spec_ds_lib, description='S.D. - Complete data').df_data

df_na = pd.concat([spec_ds_lib['df_mc_na'].df_data,\
  spec_ds_lib['df_gv_na'].df_data,spec_ds_lib['df_nv_na'].df_data,spec_ds_lib['df_sa_na'].df_data,\
  spec_ds_lib['df_ta_na'].df_data,spec_ds_lib['df_ts_na'].df_data,spec_ds_lib['df_fi_na'].df_data,\
  spec_ds_lib['df_gc_na'].df_data,spec_ds_lib['df_ci_na'].df_data\
  ],axis=1)
  
df_only_gen    =         DataSet(gen_anat_ds, 'df_only_gen_na', dataset_lib =  spec_ds_lib, description='Only gen. anat. data').df_data
df_na_without_gen =      DataSet(df_na, 'df_na_wo_gen', dataset_lib  =  spec_ds_lib, description='Whole NA normalized w/o gen. anat. data').df_data
df_na_with_gen =         DataSet(pd.concat([df_na,gen_anat_ds],axis=1), 'df_na_inc_gen', dataset_lib =  spec_ds_lib, description='Whole NA normalized with gen. anat. data').df_data

#spec_ds_lib['df_na'].df_data = normalizing_func(spec_ds_lib['df_gv_na'].df_data)


new_cols_ta = [i[:-15] for i in list(spec_ds_lib['df_ta_na'].df_data.columns)]
temp_ta = spec_ds_lib['df_ta_na'].df_data.copy()
temp_ta.columns = new_cols_ta

new_cols_sa = [i[:-15] for i in list(spec_ds_lib['df_sa_na'].df_data.columns)]
temp_sa = spec_ds_lib['df_sa_na'].df_data.copy()
temp_sa.columns = new_cols_sa

df_mul_ta_sa_na_normed = DataSet(temp_sa*temp_ta, 'df_mul_tsa_normed_na', dataset_lib=               spec_ds_lib, description='Mult. between ta and sa after norm').df_data
del temp_sa
del temp_ta
#df_mul_ta_sa_na=        DataSet(our_data_folder, 'df_mul_ta_sa_na', dataset_lib=        spec_ds_lib, description='ThickAvg * SurfArea').df_data
#spec_ds_lib['df_gv_na'].df_data = normalizing_func(spec_ds_lib['df_gv_na'].df_data)

#df_mul_ta_sa_na         -> oJr! Area * thickness for each area  

#Derived from anatomical data
df_gvsum_na=            DataSet(pd.DataFrame(df_gv_na.transpose().sum(), \
                                                     columns=['GV_total']), 'df_gvsum_na', dataset_lib=derived_ds_lib).df_data

df_tamean_na=           DataSet(pd.DataFrame([df_ta_na.transpose().mean(),\
                                    df_ta_na.transpose().median()], index=['CTA_mean',\
                                    'CTA_median']).transpose(),'df_tamean_na',dataset_lib=derived_ds_lib).df_data

'''
df_gen_on_spec = DataSet(pd.concat([df_tamean_na,df_gvsum_na, \
                        df_fcmorg_calc,df_fcm_pos_vs_neg_calc,\
                        df_fts_ints,df_gen],axis=1), 'df_gen_on_spec',\
                        dataset_lib=derived_ds_lib).df_data
'''
#Extracted from functionnal data
#Need list of regions first
ID=os.listdir(func_folder)[0]
d_fcm=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_sign_CorrMatrix_z.txt'), sep=" ", header=None)
d_cod=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None)
regionwise_cod_ds, regionwise_fcm_ds=dict(), dict()

'''
regions = list(d_cod.columns)
for i,region in enumerate(regions) :
    regionwise_cod_ds[region]   =   DataSet(our_data_folder, 'df_cod_region'+str(region), dataset_lib=spec_ds_lib,
      description='Cod - Vector Region '+str(i+1)).df_data

regions = list(d_fcm.columns)
for i, region in enumerate(regions) :
    regionwise_fcm_ds[region]   =   DataSet(our_data_folder, 'df_fcm_region_'+str(region), dataset_lib=spec_ds_lib, description='FCM - Vector Region '+str(i+1)).df_data

df_gv_na_test_2 =            DataSet(our_data_folder, 'df_gv_na', dataset_lib=               test_ds_lib_2).df_data
df_gv_na_test=               DataSet(our_data_folder, 'df_gv_na', dataset_lib=               test_ds_lib).df_data
df_mul_ta_sa_na_test=        DataSet(our_data_folder, 'df_mul_ta_sa_na', dataset_lib=        test_ds_lib).df_data
regionwise_fcm_ds_test=      DataSet(our_data_folder, 'df_fcm_region_1', dataset_lib=        test_ds_lib).df_data
regionwise_cod_ds_test   =   DataSet(our_data_folder, 'df_cod_region1', dataset_lib =        test_ds_lib).df_data
'''