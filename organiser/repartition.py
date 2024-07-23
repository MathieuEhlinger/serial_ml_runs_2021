# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:35:24 2019

@author: Mathieu
"""

import pandas as pd
import numpy as np
import copy
from sklearn.utils import shuffle
from copy import deepcopy

import math
import scipy
from scipy.stats import binom, hypergeom
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

mp1 = [['Age', float], ['glob_vol' ,float], ['Gender', bool]]
mp2 = [['Gender', bool], ['glob_vol', float], ['Age', float]]
mp1na = [['Gender', bool], ['Age', float]]

db1 = [['Gender', bool], ['Age', float]]    

def organize_after_bdi(df_org, rand=23, number_of_high=0.5):
    df = df_org.copy()
    df_low = df.loc[df['SCOREBDI']<=9]
    df_mid = df.loc[(df['SCOREBDI']>=10)]
    df_mid = df_mid.loc[(df['SCOREBDI']<=18)]
    df_high = df.loc[df['SCOREBDI']>=19]
    
    df_high=df_high.sample(frac=number_of_high, random_state=rand)
    
    number_high = df_high.shape[0]
    df_low = df_low.sample(n=number_high, random_state=rand)
    df_mid = df_mid.sample(n=number_high*2, random_state=rand)
    return pd.concat([df_high, df_low, df_mid], axis=0)

def create_2_lmdf(df_org, rand=23, number_of_high_1=0.5):
    df = df_org.copy()
    df_1 = organize_after_bdi(df, rand=rand,number_of_high=number_of_high_1)
    df = df.drop(df_1.index,axis=0)
    df_2 = organize_after_bdi(df, rand=rand, number_of_high=1)
    return shuffle(df_1,random_state=rand) , shuffle(df_2,random_state=rand)

def match(patient, df, mp, inter=0.05):
#    print('Matching ',mp[0],' ...')
    if mp[1]==float:
        df_inter=pd.DataFrame()
        inter_h=copy.deepcopy(inter)
        while df_inter.shape[0]<=3:
            plus_limit = (inter_h*patient[mp[0]])+patient[mp[0]]
            minus_limit = patient[mp[0]]-(inter_h*patient[mp[0]])
            #print('Looking for ', mp[0],' between ',\
            #      minus_limit,' and ',plus_limit)
            #print(df.shape[0])
        #final_df = final_df.divide(i).sort_values(by='balanced accuracy',ascending=False)
            df_inter = df.loc[df[mp[0]]<plus_limit]
            #print(df.loc[df[mp[0]]<plus_limit])
            #print(df_inter.shape[0])
            df_inter = df_inter.loc[df[mp[0]]>minus_limit]
            #print(df_inter.shape[0],' patients founds')
            inter_h+=0.05
            
            if inter_h>=3:
                print('Too few samples left to match on ',mp[0],' - \
                      sampling aborted for that feature')
                return df
#        print(df_inter.shape,mp)
        return df_inter
    
    elif mp[1]==bool:
#        print('Looking for ', mp[0],' equals ',\
#                  patient[mp[0]])
        
#        print((df.loc[df[mp[0]]==patient[mp[0]]]).shape)
        if (df.loc[df[mp[0]]==patient[mp[0]]]).shape[0]<=2:
            print('Too few samples left to match on ',mp[0],' - \
                      sampling aborted for that feature')
            return df
        
        return df.loc[df[mp[0]]==patient[mp[0]]]

def matchcursion(patient,df, mp, inter=0.05):
    if(len(mp))==1:
#        print(match(patient,df, mp[0]))
        return match(patient,df, mp[0],inter=inter)
    
    else :
        return match(patient, matchcursion(patient,df, mp[:-1]), mp[-1],\
                     inter=inter)



def match_for_age_and_gender(df_org, mp, rand=23, number_of_high=0.5, inter=0.05,
                             nb_mid_th=2, nb_low_th=1):
    df = df_org.copy()
    df_high = df.loc[df['SCOREBDI']>=19]
    df_high=df_high.sample(frac=number_of_high, random_state=rand)
    
    final_df=pd.DataFrame()
    for patient_index in list(df_high.index):
        
        saved_patients=pd.DataFrame()
        
        df_mid = df.loc[(df['SCOREBDI']>=10)]
        df_mid = df_mid.loc[(df_mid['SCOREBDI']<=19)]
        df_low = df.loc[df['SCOREBDI']<=9]
        df_low = df_low.loc[df_low['SCOREBDI']>=0]
        
        if nb_mid_th>0: mid_under_df = matchcursion(df_high.loc[patient_index],df_mid, mp, inter=inter)
        else : mid_under_df=pd.DataFrame()
        if nb_low_th>0: low_under_df = matchcursion(df_high.loc[patient_index],df_low, mp, inter=inter)
        else : low_under_df=pd.DataFrame()

        if nb_mid_th>0 and nb_low_th>0 :saved_patients = pd.concat([mid_under_df.sample\
                (n=nb_mid_th, random_state=rand),\
                low_under_df.sample(n=nb_low_th, random_state=rand)], axis=0)
        elif nb_mid_th==0 and nb_low_th>0:
                saved_patients= low_under_df.sample(n=nb_low_th, random_state=rand)
        elif nb_mid_th>0 and nb_low_th==0:
                saved_patients= mid_under_df.sample(n=nb_mid_th, random_state=rand)
        df = df.drop(saved_patients.index)
        final_df=pd.concat([final_df,saved_patients],axis=0)
#        print('Left sample : ',df.shape[0])
        
    final_df=pd.concat([final_df,df_high],axis=0)
    return final_df

    
def create_2_hmdf(df_org,mp, rand=23, number_of_high_1=0.5, inter=0.05,
                             nb_mid_th=2, nb_low_th=1):
    df = df_org.copy()
    df_1 = match_for_age_and_gender(df,mp, rand=rand, number_of_high=number_of_high_1, inter=inter,
                             nb_mid_th=nb_mid_th, nb_low_th=nb_low_th)
    df = df.drop(df_1.index,axis=0)
    df_2 = match_for_age_and_gender(df,mp, rand=rand, number_of_high=1, inter=inter,
                             nb_mid_th=nb_mid_th, nb_low_th=nb_low_th)
    return shuffle(df_1,random_state=rand), shuffle(df_2,random_state=rand)

def split_in_matching_datasets(df_o, parameters=db1, rs=23):
    #Used to divide one dataset in two matching datasets using an array of subarray with types
    #ex : [['Age',float],['Gender',bool]]
    
    df = copy.deepcopy(df_o)
    df = df.sample(frac=1,random_state=rs)
    final_df_1, final_df_2 = pd.DataFrame(), pd.DataFrame()
    
    while not df.empty :
      if df.shape[0]==1 :
        #
        patient = copy.deepcopy(df.iloc[0])
        final_df_1=final_df_1.append(copy.deepcopy(patient))
        df = df.drop(patient.name)
      else :
        patient = copy.deepcopy(df.iloc[0])
        final_df_1=final_df_1.append(copy.deepcopy(df.iloc[0]))
        df = df.drop(patient.name)
        second_patient = matchcursion(patient,df, parameters, inter=0.05).iloc[0]
        final_df_2=final_df_2.append(copy.deepcopy(second_patient))
        df = df.drop(second_patient.name)
    return final_df_1, final_df_2

def propensity_score_matching(df, features=['group_BIDS','PatientSex']):
    n_feat_1_pos = len(df[df[features[0]]==1])
    n_feat_pos_double = len(df[(df[features[0]]==1) & (df[features[1]]==1)])
    e_feat_1_pos = n_feat_pos_double/n_feat_1_pos
    
    n_feat_1_neg = len(df[df[features[0]]==0])
    n_feat_1_neg_feat_2_pos = len(df[(df[features[0]]==0) & (df[features[1]]==1)])
    e_feat_1_neg = n_feat_1_neg_feat_2_pos/n_feat_1_neg
    
    df["propensity"] = df[features[0]]*e_feat_1_pos + (1-df[features[0]])*e_feat_1_neg
    
    df_feat_2_pos = df[df[features[1]]==1]
    df_feat_2_neg = df[df[features[1]]==0]
    
    #Gotta find out which of the datasets is the shortest
    df_min_size=min(df_feat_2_pos.shape[0],df_feat_2_neg.shape[0])
    long_temp_df=0
    short_temp_df=0
    if df_feat_2_pos.shape[0]==df_min_size: 
      short_temp_df = deepcopy(df_feat_2_pos)
      long_temp_df = deepcopy(df_feat_2_neg)
      for_save_temp_df = deepcopy(df_feat_2_neg)
    else : 
      long_temp_df = deepcopy(df_feat_2_pos)
      short_temp_df = deepcopy(df_feat_2_neg)
      for_save_temp_df = deepcopy(df_feat_2_pos)
    
    matched_control = []
    for patient in list(short_temp_df.index):
      control_patient =long_temp_df[long_temp_df["propensity"]==short_temp_df.loc[patient]["propensity"]].sample().iloc[0]
      long_temp_df=long_temp_df.drop(control_patient.name)
      matched_control.append(control_patient.name)
    
    matched_control_df = for_save_temp_df.loc[matched_control]
    
    return pd.concat([matched_control_df,short_temp_df]).drop(['propensity'],axis=1)
'''
def df_after_iqr_calculations(df):
    df['mean'] = df.mean(axis=1)
    
    # Calculate 3 Std +/- mean upper and lower limit (IQR=2.698) 
    upperlimit = df['mean'].mean() + (df['mean'].std()*2.698)
    lowerlimit = df['mean'].mean() - (df['mean'].std()*2.698)
    
    # remove values below lowerlimit and above upperlimit 
    df_out = df.loc[(df["mean"] > lowerlimit) & (df["mean"] < upperlimit)]
    
    # store index of Ps, who are marked as outliers 
    index_low_out = df.loc[(df["mean"] < lowerlimit)].index.tolist()
    index_high_out = df.loc[(df["mean"] > upperlimit)].index.tolist()
    
    return df_out, index_high_out, index_low_out 
'''
def df_after_iqr_calculations(df_org):
    #quality control calculation function
    df=deepcopy(df_org)
    df['mean'] = df.mean(axis=1)
    
    # Calculate 3 Std +/- mean upper and lower limit (IQR=2.698) 
    upperlimit = df['mean'].mean() + (df['mean'].std()*2.698)
    lowerlimit = df['mean'].mean() - (df['mean'].std()*2.698)
    
    # remove values below lowerlimit and above upperlimit 
    df_out = df.loc[(df["mean"] > lowerlimit) & (df["mean"] < upperlimit)]
    
    # store index of Ps, who are marked as outliers 
    index_low_out = df.loc[(df["mean"] < lowerlimit)].index.tolist()
    index_high_out = df.loc[(df["mean"] > upperlimit)].index.tolist()
    df=df.drop('mean',axis=1)
    
    return df_out, index_high_out, index_low_out

def prop_match(groups, propensity, caliper = 0.05):
    ''' 
    Inputs:
    groups = Treatment assignments.  Must be 2 groups
    propensity = Propensity scores for each observation. Propensity and groups should be in the same order (matching indices)
    caliper = Maximum difference in matched propensity scores. For now, this is a caliper on the raw
            propensity; Austin reccommends using a caliper on the logit propensity.
    
    Output:
    A series containing the individuals in the control group matched to the treatment group.
    Note that with caliper matching, not every treated individual may have a match.
    '''
    
    # Check inputs
    if any(propensity <=0) or any(propensity >=1):
        raise ValueError('Propensity scores must be between 0 and 1')
    elif not(0<caliper<1):
        raise ValueError('Caliper must be between 0 and 1')
    elif len(groups)!= len(propensity):
        raise ValueError('groups and propensity scores must be same dimension')
    elif len(groups.unique()) != 2:
        raise ValueError('wrong number of groups')
        
        
    # Code groups as 0 and 1
    groups = groups == groups.unique()[0]
    N = len(groups)
    N1 = groups.sum(); N2 = N-N1
    g1, g2 = propensity[groups == 1], (propensity[groups == 0])
    # Check if treatment groups got flipped - treatment (coded 1) should be the smaller
    if N1 > N2:
       N1, N2, g1, g2 = N2, N1, g2, g1 
        
        
    # Randomly permute the smaller group to get order for matching
    morder = shuffle(list(g1.index))
    matches = pd.Series(index=g1.index)
    
    for m in morder:
        dist = abs(g1[m] - g2)
        min_index = dist.loc[dist==dist.min()].index[0]
        if dist.min() <= caliper:
            #if len(min_index)>1:  raise ValueError('More than one index would be selected - Shouldn\'t be happening. Take a look')
            matches[m] = int(min_index)
            g2 = g2.drop(min_index)
            
    return matches
#stuff = prop_match(rematched_df.BIDS, rematched_df.Propensity)

import statsmodels.api as sm
from scipy.stats import pearsonr

def residuals_for_x(x,y):  
  #add constant to predictor variables
  x = sm.add_constant(x)
  
  #fit linear regression model
  model = sm.OLS(y, x).fit() 
  
  influence = model.get_influence()
  
  #obtain standardized residuals
  standardized_residuals = influence.resid_studentized_internal
  
  #display standardized residuals
  #print(standardized_residuals)
  
  return standardized_residuals