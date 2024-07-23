# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:07:57 2018

@author: Mathieu
"""

import pandas as pd
import sys
sys.path.append('./../datai')
sys.path.append('./../pablo')
sys.path.append('./../ml_run_lib')
from datai.ds_filters import ds_2CAT
from datai.ds_filters import ds_4CAT
from ml_run_lib.transformers import apply_transformers
from ml_run_lib.transformers import nichts
from datai.data_loader import load_data
from sklearn.utils import shuffle


def categorized(df, s_filter = ds_2CAT, th =19):
    """
    Returns a preprocessed Dataframe with target score
    """
    
    result = df.copy()
    result['ds'] = df.SCOREBDI.apply(s_filter, threshold=th)
    
    return result.dropna()

def data_preprocessing(name='test', s_filter=ds_2CAT, 
            drop=['SCOREBDI'], data_choice = 'cv', 
            normalization=False ,clean= False, th=19,
            factor = 3,dynamic_drop=False,ban_thresh = 10, target = 'ds',
            gender = None, age = None, undersample = None, rand = 23, func_drop_list=[[nichts,[]]] ):
    
    """
    First, loads one of the dataset ( adds ds if target = 'ds' )
    Seconds, one-hot-encode the gender
    Third, undersample following these rules :
        * gender : Keeps the gender in input (0 : male ; 1 female)
        * age    : Keeps ages in the given range (E.G. [60,65])
        * undersample : keeps only amount of ds-patients * undersample
    Fourth : Apply series of [transformer__func, [columns not to be modifier]]
    Last, returns the result without the drop
    
    """
    
    if target == 'ds' : 
        df = categorized(load_data(data_choice),s_filter=s_filter, th=th).copy()
    else :
        df = load_data(data_choice).copy()
    
    if target !='Gender' and (data_choice in ['cv','ct','both']):
        df['Gender']= df.Gender.apply(lambda x: 1 if x=='Female' else 0)
        
    if target !='Gender' and (data_choice in ['rl_cv','rl_ct','rl_both']):
        df['Gender']= df.Gender.apply(lambda x: (x-1))
    
    #For matching purposes
    if gender   != None :
        df = df.loc[(df['Gender'] == gender)]
    if age      != None :
        df = df.loc[(df['Age'] < age[1]) & (df['Age'] >= age[0])]
    if undersample !=None :
        assert type(undersample) ==int, "Undersample should be an integer"
        try :
            dfd = df.loc[(df['ds'] == ('Depressive'))]
            dfnd= df.loc[(df['ds'] == ('Not depressive'))]
            
        except TypeError :
            dfd = df.loc[(df['ds'] == 1)]
            dfnd= df.loc[(df['ds'] == 0)]
            
        dfnd = dfnd.sample(n=(undersample*dfd.shape[0]), random_state=rand)
#                .reset_index(drop=True)
#        print(dfd.index)
        df = pd.concat([dfd, dfnd])
        df = df = shuffle(df, random_state=rand)
    
    df = apply_transformers(df,func_drop_list=func_drop_list)

#    print('Sample size :',df.shape[0])
    
#    try :
#        print('D. patients :',df.loc[(df['ds'] == ('Depressive'))].shape[0])
#    except TypeError :
#        print('D. patients :',df.loc[(df['ds'] == 1)].shape[0])
#    except KeyError :
#        print('No threshold entered')

    return df.drop(drop,axis=1)