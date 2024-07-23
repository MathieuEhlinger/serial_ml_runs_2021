import sys
sys.path.append('./../datai')
sys.path.append('./../pablo')

import pandas as pd
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from ml_run_lib.transformers import nichts

from datai.ds_filters import ds_2CAT
from datai.ds_filters import ds_4CAT
from datai.preprocessing import data_preprocessing


def scatter_plot_features(df,title='Plot_title',name='test',f1="Age",f2="rh_supramarginal_volume", other='ds'):
    """
    Plots two features as a scatterplot
    """
    print(f2)
    sns.set_style("whitegrid")
    temp_df=pd.concat([df[f1],df[f2],df[other]],axis=1)
    plot = sns.scatterplot(data=temp_df,x=f1,y=f2,hue=other)
    plot.set_title(title)
    plot=plot.get_figure()
    plot.savefig(name+'/'+'sp_'+f2+'_'+f1+'.png')
    plt.clf()

#    for cat,ax in zip(CHESS_CATS,axes.reshape(-1)):
#        x_field = f1+"_"+cat.lower()
#        y_field = f2+"_"+cat.lower()
#        tmp_df = df.loc[(~(pd.isnull(df[x_field]))) \
#                & (~(pd.isnull(df[y_field])))]
#    sns.scatterplot(data=df,x=f1,y=f2,ax=ax)
#    plt.show()



def plot_n_curves(df,cat="Rapid",n=10,same_start_date=True,min_days_played=100,
        elo_range=(500,3000)):
    """
    Plots n ranks curves randomly sampled
    :param same_start_date: Plots every curve from "zero"
    :param min_days_played: Plot players that played n days min
    :param elo_range: the min elo and max elo to draw
    """
    sub_df = df.loc[df["nb_played_days_"+cat.lower()] > min_days_played]
    sub_df = get_players_between_elo_range(sub_df,elo_range,cat)
    if sub_df.shape[0] > n:
        sub_df = sub_df.sample(n=n)
    else:
        if sub_df.shape[0] == 0:
            raise ValueError("No players found")
    flatten_data = None
    for i in range(0,n):
        user = sub_df.iloc[i]["user"]
        ranks_list = sub_df.iloc[i][cat]
        ranks_list = [(*i,user) for i in ranks_list]
        tmp_df = pd.DataFrame(ranks_list,columns=["date","rank","user"])
        tmp_df["date"] = tmp_df.date.apply(lambda x : x - min(tmp_df.date))
        if flatten_data is not None:
            flatten_data = flatten_data.append(tmp_df)
        else:
            flatten_data = tmp_df
    flatten_data["date"] = pd.to_numeric(flatten_data.date)
    flatten_data["rank"] = pd.to_numeric(flatten_data["rank"])
    sns.set_style("whitegrid")
    sns.lineplot(data=flatten_data,hue="user",x="date",y="rank",linewidth=3,\
            legend=False)
    plt.show()

def get_players_between_elo_range(df,elo_range,cat):
    return df.loc[(df["last_rank_"+cat.lower()] > elo_range[0]) &
            (df["last_rank_"+cat.lower()] < elo_range[1])]

def a_sl_of_plot(f1='Age',name='plots', s_filter=ds_2CAT, 
            drop=['SCOREBDI'], data_choice = 'cv', 
            normalization=False ,clean= False, th=19,
            factor = 3,dynamic_drop=False,ban_thresh = 10, target = 'ds',
            gender = None, age = None, undersample = None, rand = 23, func_drop_list=[[nichts,[]]] ):
    
    name = name +'_'+ data_choice
    if not os.path.exists(name):
        os.makedirs(name)
    df = data_preprocessing(name=name, s_filter=s_filter, 
                            drop=drop, data_choice = data_choice, 
                            normalization=False ,clean= clean, th=th,
                            factor = factor, dynamic_drop=dynamic_drop,
                            ban_thresh = ban_thresh, gender = gender, age =age,
                            undersample = undersample, func_drop_list=func_drop_list)
    df.describe.to_csv
    for feature in list(df.columns):
        try :   scatter_plot_features(df,name=name, f1=f1,f2=feature)
        except TypeError : print('Error for : ', feature)

#datas=['cv','rl_cv']
#all_descriptions=list()
#
#for data in datas:
#    all_descriptions.append(data_preprocessing(name='test', s_filter=ds_2CAT, 
#            drop=['SCOREBDI'], data_choice = data, 
#            normalization=False ,clean= False, th=19,
#            factor = 3,dynamic_drop=False,ban_thresh = 10, target = 'ds',
#            gender = None, age = None, undersample = None, rand = 23, 
#            func_drop_list=[[nichts,[]]]).describe())
#pd.concat(all_descriptions, axis=0, sort=True).to_csv("All_stats.csv")
