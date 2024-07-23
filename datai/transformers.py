# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:27:36 2018

@author: Mathieu
"""

import pandas as pd
import sklearn

#   [[add_total, []], [drop_all, ['sum', 'Age', 'Gender','ds']]]
#   [[add_total, []], [l1_normalize, ['sum', 'Age', 'Gender','ds']]]

#   [[add_total, []], [(lambda x,y : normalize(x,y,mult=True)), \
#   ['sum', 'Age', 'Gender','ds']]]

#----------------------------------

def nichts(df, drop=[]):
    return df.drop(drop, axis =1 ) 

def drop_all(df, drop=[]):
    return pd.DataFrame()

def add_total(df_o,drop=[], feature_name='sum',name_key='volume'):
    f_keys,df = list(),df_o.copy()
    keys =list(df.drop(drop,axis=1).columns)
    for name in keys :
        if name_key in name :
            f_keys.append(name)
    df[feature_name]=df[f_keys].sum(axis=1)
    if f_keys==[]:
        df= add_total(df_o,drop=[], feature_name='sum',name_key='thickness')
    return df

def normalize(df_o, drop=[], norm='l1',mult=False):
    df = df_o.copy()
    index, columns = df_o.index, df_o.drop(drop, axis=1).columns
    if mult : df = pd.DataFrame((sklearn.preprocessing.normalize(df.drop(drop, axis=1), \
                    norm='l1'))*1000, index=index, columns=columns)
    else : df = pd.DataFrame(sklearn.preprocessing.normalize(df.drop(drop, axis=1), \
                    norm='l1'), index=index, columns=columns)
    return df
    

#----------------------------------

def apply_transformers(df_o, func_drop_list=[[nichts,[]],[nichts,[]]]):
    df=df_o.copy()
    for func_drop in func_drop_list :
        copy = df.copy()
#        print(copy.columns == df.columns)
        df = func_drop[0](df, func_drop[1])
        
        for col in func_drop[1] :
            df[func_drop[1]]=copy[func_drop[1]]
        
        if len(df.columns)==len(copy.columns):
            
            try :       df = df[copy.columns]
            except KeyError:    pass
            except : raise
        
    return df