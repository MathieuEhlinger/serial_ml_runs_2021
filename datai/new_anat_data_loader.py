# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:21:53 2019

@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""

import pandas as pd
import copy

def one_patient_df_to_serie(df_o):
    dfop=df_o.copy()
    struct_values=dict()
    new_struct_values_glob=dict()
    for struct in list(dfop.StructName):
#        print('--------------')
#        print(struct)
        struct_values[struct]=(dfop.loc[dfop['StructName']==struct].drop(['ID','StructName'],axis=1)).to_dict()
#        print(struct_values[struct])
    # we have a dict of dicts of dicts now,  each region is first key, characteristic is second key, third key is garbage
    # Only one key, allows access to the characteristic
    #struct_values[list(struct_values.keys())[0]]
    
    for struct in struct_values.keys():
#        print('--------------')
#        print(struct)
        new_struct_values=dict()        
        for key in struct_values[struct].keys():
            new_struct_values[key]=struct_values[struct][key][list(struct_values[struct][key].keys())[0]]
    #            print(key, struct_values[new_dict][key][list(struct_values[new_dict][key].keys())[0]])
            new_struct_values_glob[struct]=copy.deepcopy(new_struct_values)
#        print(new_struct_values_glob[struct])
    
    
    last_struct_values_glob=dict()
    for key in new_struct_values_glob.keys():
#        print('---------------')
#        print(key)
        new_dict = new_struct_values_glob[key]
        second_new_dict=dict()
        for under_key in new_struct_values_glob[key].keys():
            new_key = str(key+'_'+under_key)
#            print('Key : ',new_key,' = ',new_struct_values_glob[key][under_key])
            second_new_dict[new_key] = new_dict[under_key]
    #    new_struct_values_glob[key]=copy.deepcopy(new_dict)
        last_struct_values_glob[key]=copy.deepcopy(second_new_dict)
#        print(last_struct_values_glob[key])
    #
    final_serie=pd.Series()
    i=0
    #
    for dict_key in last_struct_values_glob.keys():
#        print(i,' : ',len(last_struct_values_glob[dict_key]))
        i+=1
        final_serie=final_serie.append(pd.Series(last_struct_values_glob[dict_key]))
    return final_serie

def make_df_flat(df):
    new_df = dict()
    for patient_id in df['ID'].unique():
#        print('Patient ',patient_id,' going on ...')
        try :
            new_df[patient_id]=one_patient_df_to_serie(df.loc[df['ID']==patient_id]).rename(patient_id)
        except :
            pass
    return new_df

df = pd.read_excel('mris_anatomical_stats.xlsx')
#
#list_ID=list(df['ID'].unique())
#dfop=df.loc[df['ID']==df['ID'].unique()[0]]

def make_the_csv(df):
    flat_df = make_df_flat(df)
    try_this_df=pd.DataFrame(flat_df)
    ds_ct=pd.DataFrame.from_csv('1000BRAINS_BDI_Score_CT.csv')
    pd.concat([ds_ct[['Age','Gender','SCOREBDI']],try_this_df],1,sort=True).to_csv('1000RAINS_new.csv')
