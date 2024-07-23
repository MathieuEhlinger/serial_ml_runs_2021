from collections import OrderedDict
import pandas as pd
import os
from datai.ds_building_functions import DSLibrary, DataSet
import copy

TC_Brains_lib=                      DSLibrary('BiDirect original values')
TC_Network_lib=                     DSLibrary('BiDirect 7 Networks')
TC_residuals_Brains_lib=            DSLibrary('BiDirect residual values')
TC_residuals_c_Brains_lib=          DSLibrary('BiDirect residual values complemented')
informative_ds_lib=                 DSLibrary('General information DS Library BiDIrect')
TC_comparison_residuals_Brains_lib= DSLibrary('BiDirect residual values for comparison')

data_folder = 'input_path_to_csvs'

df_gen= DataSet(data_folder, 'BiDirect_Rematched_General_Information', informative_ds_lib).df_data
global_target= 'BIDS'
#global_target= 'Age'

TC_Brains_lib.inf_ds      =     copy.deepcopy(df_gen)
TC_residuals_Brains_lib.inf_ds      =     copy.deepcopy(df_gen)
TC_residuals_c_Brains_lib.inf_ds      =     copy.deepcopy(df_gen)
TC_comparison_residuals_Brains_lib.inf_ds =      copy.deepcopy(df_gen)
TC_Network_lib.inf_ds=                     copy.deepcopy(df_gen)

TC_Brains_lib.target_ds = copy.deepcopy(df_gen[[global_target]])
TC_residuals_Brains_lib.target_ds = copy.deepcopy(df_gen[[global_target]])
TC_residuals_c_Brains_lib.target_ds = copy.deepcopy(df_gen[[global_target]])
TC_comparison_residuals_Brains_lib.target_ds = copy.deepcopy(df_gen[[global_target]])
TC_Network_lib.target_ds = copy.deepcopy(df_gen[[global_target]])
#----------------------------
#Preparing original values DSLibrary
df_gv_na=               DataSet(data_folder, 'BiDirect_New_Matched_Grey_Matter_Volume', dataset_lib=               TC_Brains_lib, description='Gray matter volume').df_data
df_ta_na=               DataSet(data_folder, 'BiDirect_New_Matched_Cortical_Thickness', dataset_lib=               TC_Brains_lib, description='Cortical thickness average').df_data
df_sa_na=               DataSet(data_folder, 'BiDirect_New_Matched_Surface_Area', dataset_lib=                     TC_Brains_lib, description='Surface area').df_data
df_na =                 DataSet(data_folder, 'BiDirect_New_Matched_All_Structural'   , dataset_lib=                TC_Brains_lib, description='Anatomical data').df_data
df_cf =                 DataSet(data_folder, 'BiDirect_New_Matched_Confounders'   , dataset_lib=                TC_Brains_lib, description='Confounders').df_data

df_na = pd.concat([
  TC_Brains_lib['BiDirect_New_Matched_Grey_Matter_Volume'].df_data,TC_Brains_lib['BiDirect_New_Matched_Surface_Area'].df_data,\
  TC_Brains_lib['BiDirect_New_Matched_Cortical_Thickness'].df_data],axis=1)


#----------------------------
#Preparing residual values DSLibrary
df_gv_na_r=               DataSet(data_folder, 'BiDirect_New_Matched_Grey_Matter_Volume_residuals', dataset_lib=               TC_residuals_Brains_lib, description='Gray matter volume residuals').df_data
df_ta_na_r=               DataSet(data_folder, 'BiDirect_New_Matched_Cortical_Thickness_residuals', dataset_lib=               TC_residuals_Brains_lib, description='Cortical thickness average residuals').df_data
df_sa_na_r=               DataSet(data_folder, 'BiDirect_New_Matched_Surface_Area_residuals', dataset_lib=                     TC_residuals_Brains_lib, description='Surface area residuals').df_data
df_na_r =                 DataSet(data_folder, 'BiDirect_New_Matched_All_Structural_residuals'   , dataset_lib=               TC_residuals_Brains_lib, description='Anatomical data residuals').df_data
df_cf_r =                 DataSet(data_folder, 'BiDirect_New_Matched_Confounders'   , dataset_lib=                TC_residuals_Brains_lib, description='Confounders').df_data

df_na_r = pd.concat([
  TC_residuals_Brains_lib['BiDirect_New_Matched_Grey_Matter_Volume_residuals'].df_data,TC_residuals_Brains_lib['BiDirect_New_Matched_Surface_Area_residuals'].df_data,\
  TC_residuals_Brains_lib['BiDirect_New_Matched_Cortical_Thickness_residuals'].df_data],axis=1)

#----------------------------
#DataSet for comparison
df_gv_na_com=               DataSet(data_folder, 'BiDirect_New_Matched_Grey_Matter_Volume_comparison_residuals', dataset_lib=               TC_comparison_residuals_Brains_lib, description='Gray matter volume residuals for comparison').df_data
df_ta_na_com=               DataSet(data_folder, 'BiDirect_New_Matched_Cortical_Thickness_comparison_residuals', dataset_lib=               TC_comparison_residuals_Brains_lib, description='Cortical thickness average residuals for comparison').df_data
df_sa_na_com=               DataSet(data_folder, 'BiDirect_New_Matched_Surface_Area_comparison_residuals', dataset_lib=                     TC_comparison_residuals_Brains_lib, description='Surface area residuals for comparison').df_data
df_na_com =                 DataSet(data_folder, 'BiDirect_New_Matched_All_Structural_comparison_residuals'   , dataset_lib=               TC_comparison_residuals_Brains_lib, description='Anatomical data residuals for comparison').df_data

df_na_com = pd.concat([
  TC_comparison_residuals_Brains_lib['BiDirect_New_Matched_Grey_Matter_Volume_comparison_residuals'].df_data,TC_comparison_residuals_Brains_lib['BiDirect_New_Matched_Surface_Area_comparison_residuals'].df_data,\
  TC_comparison_residuals_Brains_lib['BiDirect_New_Matched_Cortical_Thickness_comparison_residuals'].df_data],axis=1)
'''
TC_comparison_residuals_Brains_lib['BiDirect_Grey_Matter_Volume_comparison_residuals'].df_data=   df_gv_na_com.drop([global_target,'Sex'], axis=1, errors='ignore')
TC_comparison_residuals_Brains_lib[ 'BiDirect_Cortical_Thickness_comparison_residuals'].df_data=  df_ta_na_com.drop([global_target,'Sex'], axis=1, errors='ignore')
TC_comparison_residuals_Brains_lib['BiDirect_Surface_Area_comparison_residuals'].df_data=         df_sa_na_com.drop([global_target,'Sex'], axis=1, errors='ignore')
TC_comparison_residuals_Brains_lib['BiDirect_All_Structural_comparison_residuals'].df_data=       df_na_com.drop([global_target,'Sex'], axis=1, errors='ignore')
'''
#----------------------------
#DataSet for comparison

df_cont=        DataSet(data_folder, 'BiDirect Cont Network', dataset_lib=               TC_Network_lib, description='Cont Network').df_data

df_limbic=      DataSet(data_folder, 'BiDirect Limbic Network', dataset_lib=               TC_Network_lib, description='Limbic Network').df_data

df_default=     DataSet(data_folder, 'BiDirect Default Network', dataset_lib=               TC_Network_lib, description='Default Network').df_data

df_sommot=      DataSet(data_folder, 'BiDirect Somatomotor Network', dataset_lib=               TC_Network_lib, description='Somatomotor Network').df_data

df_dorsalattn=  DataSet(data_folder, 'BiDirect DorsalAttention Network', dataset_lib=               TC_Network_lib, description='DorsalAttention Network').df_data

df_ventralatt=  DataSet(data_folder, 'BiDirect VentralAttention Network', dataset_lib=               TC_Network_lib, description='VentralAttention Network').df_data

df_vis=         DataSet(data_folder, 'BiDirect Visual Network', dataset_lib=               TC_Network_lib, description='Visual Network').df_data


#----------------------------
#Preparing residual values DSLibrary + completing values
'''
df_gv_na_rc=               DataSet(data_folder, 'BiDirect_New_Matched_Grey_Matter_Volume_residuals_c', dataset_lib=               TC_residuals_c_Brains_lib, description='Gray matter volume residuals with additions').df_data
df_ta_na_rc=               DataSet(data_folder, 'BiDirect_New_Matched_Cortical_Thickness_residuals_c', dataset_lib=               TC_residuals_c_Brains_lib, description='Cortical thickness average residuals with additions').df_data
df_sa_na_rc=               DataSet(data_folder, 'BiDirect_New_Matched_Surface_Area_residuals_c', dataset_lib=                     TC_residuals_c_Brains_lib, description='Surface area residuals with additions').df_data

df_na_rc =                 DataSet(data_folder, 'BiDirect_New_Matched_All_Structural_residuals_c'   , dataset_lib=               TC_residuals_c_Brains_lib, description='Anatomical data residuals with additions').df_data

TC_residuals_c_Brains_lib['BiDirect_Grey_Matter_Volume_residuals_c'].df_data=   df_gv_na_rc.drop([global_target,'Sex'], axis=1, errors='ignore')
TC_residuals_c_Brains_lib[ 'BiDirect_Cortical_Thickness_residuals_c'].df_data=  df_ta_na_rc.drop([global_target,'Sex'], axis=1, errors='ignore')
TC_residuals_c_Brains_lib['BiDirect_Surface_Area_residuals_c'].df_data=         df_sa_na_rc.drop([global_target,'Sex'], axis=1, errors='ignore')
TC_residuals_c_Brains_lib['BiDirect_All_Structural_residuals_c'].df_data=       df_na_rc.drop([global_target,'Sex'], axis=1, errors='ignore')
'''
  