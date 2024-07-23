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

default_ds_lib=         DSLibrary('Default DS library')
spec_ds_lib=            DSLibrary('Objective DS library')
major_ds_lib=           DSLibrary('Major DS library')
short_spec_ds_lib=      DSLibrary('Objective DS library shortened')
spec_old_ds_lib=        DSLibrary('Old objective DS library')
derived_ds_lib=         DSLibrary('Derived DS Library')
informative_ds_lib=     DSLibrary('General information DS Library')
test_ds_lib=            DSLibrary('Test DS library')
test_ds_lib_2=          DSLibrary('Test DS library')

#General data, like Gender, SCOREBDI, Age.
df_gen=                 DataSet(our_data_folder, 'df_gen', informative_ds_lib).df_data
#New line was added on the 14.11.2019 and adds general Volume information
df_gen = pd.concat([df_gen, pd.DataFrame.from_csv(our_data_folder+'/general_anat_feats.csv')],axis=1)
#New line added on the 13.03.2020 and adds general Volume information
df_gen = pd.concat([df_gen, pd.DataFrame.from_csv(our_data_folder+'/cat_derivated_feats.csv')],axis=1)

df_gen = pd.concat([df_gen, )],axis=1)
global_target=          'Age'

default_ds_lib.inf_ds      =     copy.deepcopy(df_gen)       
spec_ds_lib.inf_ds         =     copy.deepcopy(df_gen)
short_spec_ds_lib.inf_ds   =     copy.deepcopy(df_gen)
spec_old_ds_lib.inf_ds     =     copy.deepcopy(df_gen)
derived_ds_lib.inf_ds      =     copy.deepcopy(df_gen) 
informative_ds_lib.inf_ds  =     copy.deepcopy(df_gen) 
test_ds_lib.inf_ds         =     copy.deepcopy(df_gen)
test_ds_lib_2.inf_ds       =     copy.deepcopy(df_gen)
major_ds_lib.inf_ds        =     copy.deepcopy(df_gen)

default_ds_lib.target_ds=         copy.deepcopy(df_gen[[global_target]])
short_spec_ds_lib.target_ds=      copy.deepcopy(df_gen[[global_target]])       
spec_ds_lib.target_ds=            copy.deepcopy(df_gen[[global_target]]) 
spec_old_ds_lib.target_ds=        copy.deepcopy(df_gen[[global_target]]) 
derived_ds_lib.target_ds=         copy.deepcopy(df_gen[[global_target]]) 
informative_ds_lib.target_ds=     copy.deepcopy(df_gen[[global_target]])
test_ds_lib.target_ds=            copy.deepcopy(df_gen[[global_target]])
test_ds_lib_2.target_ds=            copy.deepcopy(df_gen[[global_target]])
major_ds_lib.target_ds        =     copy.deepcopy(df_gen[[global_target]])

#A control dataset. Somewhat outdated.
#control_df_ct=          DataSet(our_data_folder, 'control_df_ct', dataset_lib=spec_old_ds_lib).df_data
#control_df_cv=          DataSet(our_data_folder, 'control_df_cv', dataset_lib=spec_old_ds_lib).df_data


df_bdi =                DataSet(df_gen[['SCOREBDI']], 'df_bdi', description='SCOREBDI as ints', dataset_lib=    informative_ds_lib).df_data
df_ds_10=               DataSet(our_data_folder, 'df_ds_10', description='1 if BDI>10', dataset_lib=               informative_ds_lib).df_data
df_ds_19=               DataSet(our_data_folder, 'df_ds_19', dataset_lib=               informative_ds_lib).df_data
df_ds_10_cs=            DataSet(our_data_folder, 'df_ds_10_cs', dataset_lib=            informative_ds_lib).df_data
df_ds_19_cs=            DataSet(our_data_folder, 'df_ds_19_cs', dataset_lib=            informative_ds_lib).df_data

#Connectome and functionnal data.
df_cod=                 DataSet(our_data_folder, 'df_cod_optimized', dataset_lib=                 spec_ds_lib,description= 'Cod - Complete matrix').df_data
df_fcm=                 DataSet(our_data_folder, 'df_fcm_optimized', dataset_lib=                 spec_ds_lib,description= 'FCM - Complete matrix').df_data

df_cod_fcm=   DataSet(pd.concat([df_cod,df_fcm],axis=1), 'df_cod_fcm', dataset_lib =  spec_ds_lib, description='Combined DTI and FCM').df_data

df_cod_mean=            DataSet(our_data_folder, 'df_cod_mean', dataset_lib=            spec_ds_lib,description= 'Cod - Tracts per region').df_data
#df_fts_ints=            DataSet(our_data_folder, 'df_fts_ints', dataset_lib=            spec_ds_lib,description= 'FTS - Summed activations').df_data

#df_fcmorg_calc=         DataSet(our_data_folder, 'df_fcmorg_calc', dataset_lib=         spec_ds_lib,description= 'FCM - Derived').df_data
#df_fcm_pos_vs_neg_calc= DataSet(our_data_folder, 'df_fcm_pos_vs_neg_calc', dataset_lib= spec_ds_lib,description= 'FCM - Derived + vs -').df_data
#df_fcm_pos_and_neg_calc=DataSet(our_data_folder, 'df_fcm_pos_and_neg_calc', dataset_lib= spec_ds_lib,description= 'FCM - Derived + and -').df_data

#df_fts_ints             -> Integrated time series in one value
#df_fcmorg_calc          -> Dunno. Looks pretty boring. Was it the abs value of corr ?
#df_fcm_pos_vs_neg_calc  -> Sum of neg. and pos. in corr. matrix.

#Anatomical data, each type in a separate df.
df_gv_na=               DataSet(our_data_folder, 'df_gv_na', dataset_lib=               spec_ds_lib, description='Gray Matter Volume').df_data
df_mc_na=               DataSet(our_data_folder, 'df_mc_na', dataset_lib=               spec_ds_lib, description='MeanCurv').df_data
df_nv_na=               DataSet(our_data_folder, 'df_nv_na', dataset_lib=               spec_ds_lib, description='NumVert').df_data
df_sa_na=               DataSet(our_data_folder, 'df_sa_na', dataset_lib=               spec_ds_lib, description='SurfArea').df_data
df_ta_na=               DataSet(our_data_folder, 'df_ta_na', dataset_lib=               spec_ds_lib, description='ThickAvg').df_data
df_ts_na=               DataSet(our_data_folder, 'df_ts_na', dataset_lib=               spec_ds_lib, description='ThickStd').df_data
df_fi_na=               DataSet(our_data_folder, 'df_fi_na', dataset_lib=               spec_ds_lib, description='FoldInd').df_data
df_gc_na=               DataSet(our_data_folder, 'df_gc_na', dataset_lib=               spec_ds_lib, description='GausCurv').df_data
df_ci_na=               DataSet(our_data_folder, 'df_ci_na', dataset_lib=               spec_ds_lib, description='CurvInd').df_data
df_na =                 DataSet(our_data_folder, 'df_na'   , dataset_lib=               spec_ds_lib, description='S.D. - Complete data').df_data
df_mul_ta_sa_na=        DataSet(our_data_folder, 'df_mul_ta_sa_na', dataset_lib=        spec_ds_lib, description='ThickAvg * SurfArea').df_data

#df_mul_ta_sa_na         -> oJr! Area * thickness for each area  
df_cod_na=    DataSet(pd.concat([df_cod,df_na],axis=1), 'df_cod_na', dataset_lib =  spec_ds_lib, description='Combined S.D. and DTI').df_data            
df_cod_fcm=   DataSet(pd.concat([df_cod,df_fcm],axis=1), 'df_cod_fcm', dataset_lib =  spec_ds_lib, description='Combined DTI and FCM').df_data             
df_na_fcm=    DataSet(pd.concat([df_na,df_fcm],axis=1), 'df_na_fcm', dataset_lib =  spec_ds_lib, description='Combined S.D. and FCM').df_data
df_all_three= DataSet(pd.concat([df_na,df_fcm, df_cod],axis=1), 'df_na_fcm_cod', dataset_lib =  spec_ds_lib, description='Combined S.D., FCM and DTI').df_data               
#Derived from anatomical data
df_gvsum_na=            DataSet(pd.DataFrame(df_gv_na.transpose().sum(), \
                                                     columns=['GV_total']), 'df_gvsum_na', dataset_lib=derived_ds_lib).df_data

df_tamean_na=           DataSet(pd.DataFrame([df_ta_na.transpose().mean(),\
                                    df_ta_na.transpose().median()], index=['CTA_mean',\
                                    'CTA_median']).transpose(),'df_tamean_na',dataset_lib=derived_ds_lib).df_data

'''
#################################################################################################
<Major ds only>
#################################################################################################
'''
DataSet(our_data_folder, 'df_cod_optimized', dataset_lib = major_ds_lib,description= 'Cod - Complete matrix').df_data
DataSet(our_data_folder, 'df_na'   , dataset_lib = major_ds_lib, description='S.D. - Complete data').df_data
DataSet(pd.concat([df_cod,df_na],axis=1), 'df_cod_na', dataset_lib =  major_ds_lib, description='Combined S.D. and DTI').df_data            
DataSet(our_data_folder, 'df_fcm_optimized', dataset_lib = major_ds_lib,description= 'FCM - Complete matrix').df_data
DataSet(pd.concat([df_cod,df_fcm],axis=1), 'df_cod_fcm', dataset_lib = major_ds_lib, description='Combined DTI and FCM').df_data
DataSet(pd.concat([df_na,df_fcm],axis=1), 'df_na_fcm', dataset_lib = major_ds_lib, description='Combined S.D. and FCM').df_data
DataSet(pd.concat([df_na,df_fcm, df_cod],axis=1), 'df_na_fcm_cod', dataset_lib = major_ds_lib, description='Combined S.D., FCM and DTI').df_data
DataSet(pd.concat([df_cod,df_na],axis=1), 'df_cod_na', dataset_lib = major_ds_lib, description='Combined S.D. and DTI').df_data            

'''
#################################################################################################
</Major ds only>
#################################################################################################
'''

#Here starts the ds building for each region in cod and func.
#Extracted from functionnal data
#Need list of regions first
ID=os.listdir(func_folder)[0]
d_fcm=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_sign_CorrMatrix_z.txt'), sep=" ", header=None)
d_cod=pd.read_csv(str(func_folder+'/'+ID+'/'+ID+'_dwi_connectome_zerodiagonal.txt'), sep=" ", header=None)
regionwise_cod_ds, regionwise_fcm_ds=dict(), dict()


regions = list(d_cod.columns)
for i,region in enumerate(regions) :
    regionwise_cod_ds[region]   =   DataSet(our_data_folder, 'df_cod_region'+str(region), dataset_lib=spec_ds_lib,
      description='Cod - Vector Region '+str(i+1)).df_data
'''
regions = list(d_fcm.columns)
for i, region in enumerate(regions) :
    regionwise_fcm_ds[region]   =   DataSet(our_data_folder, 'df_fcm_region_'+str(region), dataset_lib=spec_ds_lib, description='FCM - Vector Region '+str(i+1)).df_data
'''
'''
df_gv_na_test_2 =            DataSet(our_data_folder, 'df_gv_na', dataset_lib=               test_ds_lib_2).df_data
df_gv_na_test=               DataSet(our_data_folder, 'df_gv_na', dataset_lib=               test_ds_lib).df_data
df_mul_ta_sa_na_test=        DataSet(our_data_folder, 'df_mul_ta_sa_na', dataset_lib=        test_ds_lib).df_data
regionwise_fcm_ds_test=      DataSet(our_data_folder, 'df_fcm_region_1', dataset_lib=        test_ds_lib).df_data
regionwise_cod_ds_test   =   DataSet(our_data_folder, 'df_cod_region1', dataset_lib =        test_ds_lib).df_data
'''
#-------------------------------------
'''
df_gv_na_2=               DataSet(our_data_folder, 'df_gv_na', dataset_lib=               short_spec_ds_lib, description='Gray Matter Volume').df_data
df_mc_na_2=               DataSet(our_data_folder, 'df_mc_na', dataset_lib=               short_spec_ds_lib, description='MeanCurv').df_data
df_nv_na_2=               DataSet(our_data_folder, 'df_nv_na', dataset_lib=               short_spec_ds_lib, description='NumVert').df_data
df_sa_na_2=               DataSet(our_data_folder, 'df_sa_na', dataset_lib=               short_spec_ds_lib, description='SurfArea').df_data
df_ta_na_2=               DataSet(our_data_folder, 'df_ta_na', dataset_lib=               short_spec_ds_lib, description='ThickAvg').df_data
df_ts_na_2=               DataSet(our_data_folder, 'df_ts_na', dataset_lib=               short_spec_ds_lib, description='ThickStd').df_data
df_fi_na_2=               DataSet(our_data_folder, 'df_fi_na', dataset_lib=               short_spec_ds_lib, description='FoldInd').df_data
df_gc_na_2=               DataSet(our_data_folder, 'df_gc_na', dataset_lib=               short_spec_ds_lib, description='GausCurv').df_data
df_ci_na_2=               DataSet(our_data_folder, 'df_ci_na', dataset_lib=               short_spec_ds_lib, description='CurvInd').df_data
df_na_2 =                 DataSet(our_data_folder, 'df_na'   , dataset_lib=               short_spec_ds_lib, description='S.D. - Complete data').df_data
df_mul_ta_sa_na_2=        DataSet(our_data_folder, 'df_mul_ta_sa_na', dataset_lib=        short_spec_ds_lib, description='ThickAvg * SurfArea').df_data
df_cod_mean=            DataSet(our_data_folder, 'df_cod_mean', dataset_lib=              short_spec_ds_lib,description= 'Cod - Tracts per region').df_data
df_fts_ints=            DataSet(our_data_folder, 'df_fts_ints', dataset_lib=              short_spec_ds_lib,description= 'FTS - Summed activations').df_data
df_fcm=                 DataSet(our_data_folder, 'df_fcm', dataset_lib=                   short_spec_ds_lib,description= 'FCM - Complete matrix').df_data
#df_fcmorg_calc=         DataSet(our_data_folder, 'df_fcmorg_calc', dataset_lib=           spec_ds_lib,description= 'FCM - Derived').df_data
#df_fcm_pos_vs_neg_calc= DataSet(our_data_folder, 'df_fcm_pos_vs_neg_calc', dataset_lib=   spec_ds_lib,description= 'FCM - Derived + vs -').df_data
df_fcm_pos_and_neg_calc=DataSet(our_data_folder, 'df_fcm_pos_and_neg_calc', dataset_lib=  short_spec_ds_lib,description= 'FCM - Derived + and -').df_data
df_cod=                 DataSet(our_data_folder, 'df_cod', dataset_lib=                   short_spec_ds_lib,description= 'Cod - Complete matrix').df_data
'''