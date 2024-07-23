# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:10:59 2018

@author: Mathieu
"""

import pandas as pd
import os

def doubles(index):
    
    """
    Returns list of clones in a list.
    """
    
    freq = {x:list(index).count(x) for x in index}
    doubles = [x for x in freq.keys() if freq[x]>1]
    return doubles

def rl_reader_ct():
    get_cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    
    data_file_41 = "aparc_stats_T1_lh_eNKI.csv"
    data_file_42 = "aparc_stats_T1_rh_eNKI.csv"
    
    data_file_51 = "BDI_1.csv"
    data_file_52 = "BDI_2.csv"
    
    df1 = pd.read_csv(data_file_41,sep=',',index_col=False ).dropna()
    df1['ID']=df1['lh.aparc.thickness'].apply(lambda x:  (x[4:13]))
    df1= df1.drop(['lh.aparc.thickness','BrainSegVolNotVent','eTIV',\
                   'lh_MeanThickness_thickness'],\
                  axis = 1).set_index('ID')
    
    df2 = pd.read_csv(data_file_42,sep=',',index_col=False ).dropna()
    
    df2['ID']=df2['rh.aparc.thickness'].apply(lambda x:  (x[4:13]))
    df2= df2.drop(['rh.aparc.thickness','BrainSegVolNotVent','eTIV',\
                   'rh_MeanThickness_thickness'],\
                    axis = 1).set_index('ID')
    df = pd.concat([df1, df2], axis=1)
    
    attributes = pd.read_csv(data_file_51,sep=',',index_col=False ).dropna()
    attributes['ID']=attributes['Anonymized ID']
    attributes= attributes.drop(['Subject Type','Anonymized ID','Visit','R','Native language','Ethnicity'],\
                    axis = 1).set_index('ID')
    
    targ = pd.read_csv(data_file_52,sep=',',index_col=False ).dropna()
    targ['ID']=targ['Anonymized ID']
    targ = targ.set_index('ID')
    targ=targ['BDI Total']
    
    ids, df_ids, attributes_ids=[], [], []
    #First : Creating a double-free ID list of df
    for a in (set(df.index)-set(doubles(df.index))):
            df_ids.append(a)
    #Taking out doubles from : Creating a double-free ID list of df
    for a in (set(attributes.index)-set(doubles(attributes.index))):
            attributes_ids.append(a)
    # Removing all doubles from the selection
    # Reason : Doubles can't be associated with the BDI
    for a in (set(targ.index)-set(doubles(targ.index))):
        if (a in df_ids) & (a in attributes_ids):
            ids.append(a)
    

    #Checking for an unseen double (unlikely)
    assert len(ids) == len(set(ids))
    
    final_df = dict()
    final_attributes= dict()
    final_targets = dict()
    
    for ind in ids :
#           In case the patient has multiple datas available, raise error
        if type(df.loc[ind])== type(pd.DataFrame()):
            raise 'Multiple datas for one ID'
        else :
            final_df[ind] = (df.loc[ind])
            
    for ind in ids :
#           In case the patient has multiple datas available, only the first one is selected
        if type(attributes.loc[ind])== type(pd.DataFrame()):
            raise 'Multiple attributes for one ID'
        else :
            final_attributes[ind] = (attributes.loc[ind])
            
    for ind in ids :
#           In case the patient has multiple attributes available, raise error
        if type(targ.loc[ind])== type(pd.DataFrame()):
            raise 'Multiple targets for one ID'
        else :
            final_targets[ind] = (targ.loc[ind])
#        
    final_df=pd.DataFrame(final_df).transpose()
    final_attributes=pd.DataFrame(final_attributes).transpose()
    final_targets=pd.Series(final_targets)
    
    #Making sure there are only int type value in the BDI
    for val in final_targets:
        try :
            int(val)
        except ValueError:
            final_targets = final_targets.drop( \
                final_targets.loc[final_targets==val].index, axis =0)

            
    #If you feel like checking anything...        
#        final_df.to_csv('First results df.csv')
#        final_attributes.to_csv('First results Att.csv')
#        final_targets.to_csv('First results target.csv')        
    
    result = pd.concat([final_df,final_attributes], axis=1)
    result['SCOREBDI']=final_targets.apply(lambda x:int(x))
    
    os.chdir(get_cwd)
    return result


def rl_reader_cv():
    get_cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    data_file_31 = "aparc_stats_T1_lh_vol_eNKI.csv"
    data_file_32 = "aparc_stats_T1_rh_vol_eNKI.csv"
    data_file_51 = "BDI_1.csv"
    data_file_52 = "BDI_2.csv"
    
    df1 = pd.read_csv(data_file_31,sep=',',index_col=False ).dropna()
    df1['ID']=df1['lh.aparc.volume'].apply(lambda x:  (x[4:13]))
    df1= df1.drop(['lh.aparc.volume','BrainSegVolNotVent','eTIV'],\
                  axis = 1).set_index('ID')
    
    df2 = pd.read_csv(data_file_32,sep=',',index_col=False ).dropna()
    
    df2['ID']=df2['rh.aparc.volume'].apply(lambda x:  (x[4:13]))
    df2= df2.drop(['rh.aparc.volume','BrainSegVolNotVent','eTIV'],\
                    axis = 1).set_index('ID')
    df = pd.concat([df1, df2], axis=1)
    
    attributes = pd.read_csv(data_file_51,sep=',',index_col=False ).dropna()
    attributes['ID']=attributes['Anonymized ID']
    attributes= attributes.drop(['Subject Type','Anonymized ID','Visit','R','Native language','Ethnicity'],\
                    axis = 1).set_index('ID')
    
    targ = pd.read_csv(data_file_52,sep=',',index_col=False ).dropna()
    targ['ID']=targ['Anonymized ID']
    targ = targ.set_index('ID')
    targ=targ['BDI Total']
    
    ids, df_ids, attributes_ids=[], [], []
    #First : Creating a double-free ID list of df
    for a in (set(df.index)-set(doubles(df.index))):
            df_ids.append(a)
    #Taking out doubles from : Creating a double-free ID list of df
    for a in (set(attributes.index)-set(doubles(attributes.index))):
            attributes_ids.append(a)
    # Removing all doubles from the selection
    # Reason : Doubles can't be associated with the BDI
    for a in (set(targ.index)-set(doubles(targ.index))):
        if (a in df_ids) & (a in attributes_ids):
            ids.append(a)
    

    #Checking for an unseen double (unlikely)
    assert len(ids) == len(set(ids))
    
    final_df = dict()
    final_attributes= dict()
    final_targets = dict()
    
    for ind in ids :
#           In case the patient has multiple datas available, raise error
        if type(df.loc[ind])== type(pd.DataFrame()):
            raise 'Multiple datas for one ID'
        else :
            final_df[ind] = (df.loc[ind])
            
    for ind in ids :
#           In case the patient has multiple datas available, only the first one is selected
        if type(attributes.loc[ind])== type(pd.DataFrame()):
            raise 'Multiple attributes for one ID'
        else :
            final_attributes[ind] = (attributes.loc[ind])
            
    for ind in ids :
#           In case the patient has multiple attributes available, raise error
        if type(targ.loc[ind])== type(pd.DataFrame()):
            raise 'Multiple targets for one ID'
        else :
            final_targets[ind] = (targ.loc[ind])
#        
    final_df=pd.DataFrame(final_df).transpose()
    final_attributes=pd.DataFrame(final_attributes).transpose()
    final_targets=pd.Series(final_targets)
    
    #Making sure there are only int type value in the BDI
    for val in final_targets:
        try :
            int(val)
        except ValueError:
            final_targets = final_targets.drop( \
                final_targets.loc[final_targets==val].index, axis =0)

                

        #If you feel like checking anything...        
#        final_df.to_csv('First results df.csv')
#        final_attributes.to_csv('First results Att.csv')
#        final_targets.to_csv('First results target.csv')        
        
    result = pd.concat([final_df,final_attributes], axis=1)
    result['SCOREBDI']=final_targets.apply(lambda x:int(x))
    os.chdir(get_cwd)
    return result
#   return df1, df2, df, attributes, targ, ids, final_df, final_attributes, final_targets


def rl_reader_ct_age():
    get_cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    
    data_file_41 = "aparc_stats_T1_lh_eNKI.csv"
    data_file_42 = "aparc_stats_T1_rh_eNKI.csv"
    
    data_file_51 = "BDI_1.csv"
#    data_file_52 = "BDI_2.csv"
#    
    df1 = pd.read_csv(data_file_41,sep=',',index_col=False ).dropna()
    df1['ID']=df1['lh.aparc.thickness'].apply(lambda x:  (x[4:13]))
    df1= df1.drop(['lh.aparc.thickness','BrainSegVolNotVent','eTIV',\
                   'lh_MeanThickness_thickness'],\
                  axis = 1).set_index('ID')
    
    df2 = pd.read_csv(data_file_42,sep=',',index_col=False ).dropna()
    
    df2['ID']=df2['rh.aparc.thickness'].apply(lambda x:  (x[4:13]))
    df2= df2.drop(['rh.aparc.thickness','BrainSegVolNotVent','eTIV',\
                   'rh_MeanThickness_thickness'],\
                    axis = 1).set_index('ID')
    df = pd.concat([df1, df2], axis=1)
    
    attributes = pd.read_csv(data_file_51,sep=',',index_col=False ).dropna()
    attributes['ID']=attributes['Anonymized ID']
    attributes= attributes.drop(['Subject Type','Anonymized ID','Visit',\
                                 'R','Native language','Ethnicity'],\
                    axis = 1).set_index('ID')
#    
    ids, df_ids, attributes_ids=[], [], []
    df_ids = set(df.index)-set(doubles(df.index))
            
    #Taking out doubles from : Creating a double-free ID list of df
    attributes_ids = set(attributes.index)-set(doubles(attributes.index))
            
#     Removing all doubles from the selection
#     Reason : Not matching data between attributes and c. datas
    for a in df_ids:
        if (a in df_ids) & (a in attributes_ids):
            ids.append(a)
    

#    Checking for an unseen double (unlikely)
    assert len(ids) == len(set(ids))
    
    final_df = dict()
    final_attributes= dict()
#    final_targets = dict()
    
    for ind in ids :
#           In case the patient has multiple datas available, raise error
        if type(df.loc[ind])== type(pd.DataFrame()):
            raise 'Multiple datas for one ID'
        else :
            final_df[ind] = (df.loc[ind])
            
    for ind in ids :
#           In case the patient has multiple datas available, only the first one is selected
        if type(attributes.loc[ind])== type(pd.DataFrame()):
            raise 'Multiple attributes for one ID'
        else :
            final_attributes[ind] = (attributes.loc[ind])
            
#    for ind in ids :
##           In case the patient has multiple attributes available, raise error
#        if type(targ.loc[ind])== type(pd.DataFrame()):
#            raise 'Multiple targets for one ID'
#        else :
#            final_targets[ind] = (targ.loc[ind])
##        
    final_df=pd.DataFrame(final_df).transpose()
    final_attributes=pd.DataFrame(final_attributes).transpose()
#    final_targets=pd.Series(final_targets)
#    
#    #Making sure there are only int type value in the BDI
#    for val in final_targets:
#        try :
#            int(val)
#        except ValueError:
#            final_targets = final_targets.drop( \
#                final_targets.loc[final_targets==val].index, axis =0)

            
    #If you feel like checking anything...        
#        final_df.to_csv('First results df.csv')
#        final_attributes.to_csv('First results Att.csv')
#        final_targets.to_csv('First results target.csv')        
    
    result = pd.concat([final_df,final_attributes], axis=1)
#    result['SCOREBDI']=final_targets.apply(lambda x:int(x))
    os.chdir(get_cwd)
    return result

def rl_reader_cv_age():
    get_cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    
    data_file_31 = "aparc_stats_T1_lh_vol_eNKI.csv"
    data_file_32 = "aparc_stats_T1_rh_vol_eNKI.csv"
    data_file_51 = "BDI_1.csv"
    data_file_52 = "BDI_2.csv"
    
    df1 = pd.read_csv(data_file_31,sep=',',index_col=False ).dropna()
    df1['ID']=df1['lh.aparc.volume'].apply(lambda x:  (x[4:13]))
    df1= df1.drop(['lh.aparc.volume','BrainSegVolNotVent','eTIV'],\
                  axis = 1).set_index('ID')
    
    df2 = pd.read_csv(data_file_32,sep=',',index_col=False ).dropna()
    
    df2['ID']=df2['rh.aparc.volume'].apply(lambda x:  (x[4:13]))
    df2= df2.drop(['rh.aparc.volume','BrainSegVolNotVent','eTIV'],\
                    axis = 1).set_index('ID')
    df = pd.concat([df1, df2], axis=1)
    
    attributes = pd.read_csv(data_file_51,sep=',',index_col=False ).dropna()
    attributes['ID']=attributes['Anonymized ID']
    attributes= attributes.drop( \
                                ['Subject Type', 'Anonymized ID','Visit',\
                                 'R','Native language','Ethnicity'],\
                    axis = 1).set_index('ID')
    
    targ = pd.read_csv(data_file_52,sep=',',index_col=False ).dropna()
    targ['ID']=targ['Anonymized ID']
    targ = targ.set_index('ID')
    targ=targ['BDI Total']
    
    ids, df_ids, attributes_ids=[], [], []
#    First : Creating a double-free ID list of df
    df_ids = set(df.index)-set(doubles(df.index))
            
    #Taking out doubles from : Creating a double-free ID list of df
    attributes_ids = set(attributes.index)-set(doubles(attributes.index))
            
#     Removing all doubles from the selection
#     Reason : Not matching data between attributes and c. datas
    for a in df_ids:
        if (a in df_ids) & (a in attributes_ids):
            ids.append(a)
    

#    Checking for an unseen double (unlikely)
    assert len(ids) == len(set(ids))
    
    final_df = dict()
    final_attributes= dict()
#    final_targets = dict()
    
    for ind in ids :
#           In case the patient has multiple datas available, raise error
        if type(df.loc[ind])== type(pd.DataFrame()):
            raise 'Multiple datas for one ID'
        else :
            final_df[ind] = (df.loc[ind])
            
    for ind in ids :
#           In case the patient has multiple datas available, only the first one is selected
        if type(attributes.loc[ind])== type(pd.DataFrame()):
            raise 'Multiple attributes for one ID'
        else :
            final_attributes[ind] = (attributes.loc[ind])
            
#    for ind in ids :
##           In case the patient has multiple attributes available, raise error
#        if type(targ.loc[ind])== type(pd.DataFrame()):
#            raise 'Multiple targets for one ID'
#        else :
#            final_targets[ind] = (targ.loc[ind])
##        
    final_df=pd.DataFrame(final_df).transpose()
    final_attributes=pd.DataFrame(final_attributes).transpose()
#    final_targets=pd.Series(final_targets)
#    
#    #Making sure there are only int type value in the BDI
#    for val in final_targets:
#        try :
#            int(val)
#        except ValueError:
#            final_targets = final_targets.drop( \
#                final_targets.loc[final_targets==val].index, axis =0)

            
    #If you feel like checking anything...        
#        final_df.to_csv('First results df.csv')
#        final_attributes.to_csv('First results Att.csv')
#        final_targets.to_csv('First results target.csv')        
    
    result = pd.concat([final_df,final_attributes], axis=1)
#    result['SCOREBDI']=final_targets.apply(lambda x:int(x))
    os.chdir(get_cwd)
    return result


def load_data(data_choice = 'cv', target = 'ds' ):
    
    """
    Choose among the many data sets
    'cv', 'ct', 'both', 'rl_cv', 'rl_ct', 'rl_both'
    """
    get_cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    
    drt = ['ds','SCOREBDI']     # drt : list of targets for depression status
                                # Used for increasing amount of data
                                # with those of BDI-less patients
                                
    data_file_1 = "1000BRAINS_BDI_Score_Vol.csv"
    data_file_2 = "1000BRAINS_BDI_Score_CT.csv"
    
#    print('---------------------')
#    print('Input : ', data_choice,'\nTarget : ', target)
    
    df= None
    
    if data_choice == 'cv':
        df = pd.read_csv(data_file_1,sep=';',index_col =0)
        df= df.dropna(axis = 0, how='any')
        
    elif data_choice == 'ct':         
        df = pd.read_csv(data_file_2,sep=',',index_col =0).dropna(axis = 0, \
                        how='any')
        
    elif data_choice == 'both' :
        df1 = pd.read_csv(data_file_1,sep=';',  index_col =0)
        df2 = pd.read_csv(data_file_2,sep=',',  index_col =0).drop( \
                         ['Gender','Age','SCOREBDI'], axis=1)
        df = pd.concat([df1, df2], axis=1)
#        print(df.index)
#        df.to_csv(path_or_buf="check.csv", index=False)
        
    elif( data_choice == 'rl_cv' and target in drt) :
        print('what_rl_cv')
        df = rl_reader_cv()
    elif( data_choice == 'rl_cv' and target not in drt) :
        df = rl_reader_cv_age()
        df['SCOREBDI']= 0
    elif data_choice == 'rl_ct'and target in drt:
        df = rl_reader_ct()
    elif data_choice == 'rl_ct'and target not in drt:
        df = rl_reader_ct_age()
        df['SCOREBDI']= 0
        
    elif data_choice == 'rl_both' and target in drt:
        df1 = rl_reader_cv()
        df2 = rl_reader_ct()
        df = pd.concat([df1, \
                        df2.drop(['Age','Gender','SCOREBDI'],axis=1)], axis = 1)
    
    elif data_choice == 'rl_both' and target not in drt:
        df1 = rl_reader_cv_age()
        df2 = rl_reader_ct_age()
        df = pd.concat([df1,df2.drop(['Age','Gender'],axis=1)], axis = 1)
        df['SCOREBDI']= 0

    
    else :
        print(type(data_choice))
        raise NameError((str(data_choice+' not included in selection')))
    
    if target=='ds' or target=='SCOREBDI':
        df= df.drop(df.SCOREBDI.loc[df['SCOREBDI']<0].index)
    os.chdir(get_cwd)
    return df

#----------------------------------
