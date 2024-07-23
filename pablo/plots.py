# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:49:11 2018

@author: Mathieu
"""
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot
import math
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.datasets import make_spd_matrix
#import matplotlib as mpl
path_for_save='add_path_for_save'

def show_heat_map(df):
    corr = df.corr()
    sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#def unflatten(serie, new_square=84):
    

def already_cm_show_heat_map_tract_log(df_o, path, title='Results of probabilistic tractography analysis (+1)',xlabel='', ylabel=''):
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.reset_orig()
    plt.clf()
    
    #log_workaround
    #k, dirty work around. log(0) IS painful
    df = df_o + 1
    cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(df.min().min())), 1+math.ceil(math.log10(df.max().max())))]
    
    log_norm = LogNorm(vmin=df.min().min(), vmax=df.max().max())
    #grid_kws = {"height_ratios": (.9, .05), "hspace": .3} 
    #ax = plt.axes() 
    #fig, ax = plt.subplots(figsize=(15,15)) 
    grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
    #f, (ax, cbar_ax) = plt.subplots(nrows=1,ncols=2, gridspec_kw=grid_kws)
    #ax = plt.subplot()
    #cbar_ax = plt.subplot()
    #cbar_ax = fig.add_axes([.905, .3, .05, .3])
    #log_norm = LogNorm(vmin=df.min().min(), vmax=df.max().max())
    #cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(df.min().min())), 1+math.ceil(math.log10(df.max().max())))]


    #ax.set(adjustable='box-forced', aspect='equal')
    SMALL_SIZE = 5
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    TITLE_SIZE = 15

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    pyplot.figure(figsize=(15, 15))
    #sns.set(rc={'figure.figsize':(15,15)})
    #cmap=sns.dark_palette("white", n_colors=10),
    #cmap=sns.color_palette("colorblind", 10),
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=13, center="light")
    sns.set(rc={'figure.figsize':(15,15)})
    #sns.set(font_scale=1.4)
    sns_fig = sns.heatmap(df,
            xticklabels=df.columns.values,
            yticklabels=df.columns.values,
            #center=0,
            #linewidths=.5,
            cmap = cmap,
            #vmax = 3.5,
            #vmin= -3.5,
            #ax=ax,
            #cbar_ax=cbar_ax,
            #cbar_kws={"orientation": "horizontal"},
            #norm=log_norm,
            
            #cbar_kws={"ticks": cbar_ticks},
            cbar_kws={"shrink": .82, "ticks": cbar_ticks},
            square=True,
            norm=log_norm,
            )
            

    sns_fig.set_title(title, fontsize=TITLE_SIZE)
    sns_fig.set(xlabel=xlabel, ylabel=ylabel)
    #ax.set_title('Correlation matrix of brain-region fMRI-measured activity fluctuations', fontsize=TITLE_SIZE)
    sns_fig.set_xticklabels(df.columns.values,rotation=45,fontsize=SMALL_SIZE)
    sns_fig.set_yticklabels(df.columns.values,rotation=0,fontsize=SMALL_SIZE)
    #for ax in sns_fig.axes.flat:
    #sns_fig.set_ylabel(df.columns.values,fontsize=SMALL_SIZE)
    #sns_fig.set_xlabel(df.columns.values,fontsize=SMALL_SIZE)

            # set aspect of all axis
          #      ax.set_aspect('equal','box-forced')
            # set background color of axis instance
            #   ax.set_axis_bgcolor(facecolor)
    #sns_fig.set_xlabel(
    print('Save to :',path)
    sns_fig.figure.savefig(path)
    plt.close()

def line_plots(df_o,path):
    plt.clf()
    ax = sns.lineplot(data=df_o)
    ax.figure.savefig(path)
    plt.close()

def make_random_sym_matr_0_diag(dim):
    r_m = make_spd_matrix(dim)
    random_array_corrected=list()
    for i,arr in enumerate(rm):
      new_array=list()
      for second_i, value in arr :
        if i == second_i:
          new_array.append(0)
        else :
          new_array.append(value)
      random_array_corrected.append(new_array)
    return random_array_corrected

def already_cm_show_heat_map(df_o, path,title='Correlation matrix of brain-region fMRI-measured activity fluctuations', xlabel='', ylabel='',
      vmax=None, vmin=None):
    
    if vmax is None :  vmax= round(df_o.max().max())+1
    if vmin is None :  vmin= round(df_o.min().min())-1
    #log_workaround
    mpl.rcParams.update(mpl.rcParamsDefault)
    sns.reset_orig()
    #df = df_o + 1
    df = df_o
    plt.clf()
    #grid_kws = {"height_ratios": (.9, .05), "hspace": .3} 
    #ax = plt.axes() 
    #fig, ax = plt.subplots(figsize=(15,15)) 
    grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
    #f, (ax, cbar_ax) = plt.subplots(nrows=1,ncols=2, gridspec_kw=grid_kws)
    #ax = plt.subplot()
    #cbar_ax = plt.subplot()
    #cbar_ax = fig.add_axes([.905, .3, .05, .3])
    #log_norm = LogNorm(vmin=df.min().min(), vmax=df.max().max())
    #cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(df.min().min())), 1+math.ceil(math.log10(df.max().max())))]


    #ax.set(adjustable='box-forced', aspect='equal')
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    TITLE_SIZE = 15

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    pyplot.figure(figsize=(15, 15))
    #sns.set(rc={'figure.figsize':(15,15)})
    #cmap=sns.dark_palette("white", n_colors=13),
    #cmap=sns.color_palette("colorblind", 10),
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=13, center="light")
    sns.set(rc={'figure.figsize':(15,15)})
    #sns.set(font_scale=1.4)
    sns_fig = sns.heatmap(df,
            xticklabels=df.columns.values,
            yticklabels=df.columns.values,
            center=0,
            #linewidths=.5,
            cmap = cmap,
            vmax = vmax,
            vmin = vmin,
            #ax=ax,
            #cbar_ax=cbar_ax,
            #cbar_kws={"orientation": "horizontal"},
            #norm=log_norm,
            
            #cbar_kws={"ticks": cbar_ticks},
            cbar_kws={"shrink": .5},
            square=True
            )

    #sns_fig.set_title('Results of probabilistic tractography analysis', fontsize=TITLE_SIZE)
    
    sns_fig.set_title(title, fontsize=TITLE_SIZE)
    sns_fig.set(xlabel=xlabel, ylabel=ylabel)
    
    #Only one in 10 label should be displayed. Here the code making the name list
    names     = df.columns.values
    new_names = ['']*len(df.columns.values)
    for i, value in enumerate(names): 
        if i%10==0:
          new_names[i] = names[i]
            
    sns_fig.set_xticklabels(new_names,rotation=45,fontsize=SMALL_SIZE)
    sns_fig.set_yticklabels(new_names,rotation=0,fontsize=SMALL_SIZE)
    #for ax in sns_fig.axes.flat:
    #sns_fig.set_ylabel(df.columns.values,fontsize=SMALL_SIZE)
    #sns_fig.set_xlabel(df.columns.values,fontsize=SMALL_SIZE)

            # set aspect of all axis
          #      ax.set_aspect('equal','box-forced')
            # set background color of axis instance
            #   ax.set_axis_bgcolor(facecolor)
    #sns_fig.set_xlabel(
    print('Save to :',path)
    sns_fig.figure.savefig(path)
    plt.close()

def plotting_these_fcms(l_fcm, df_ds, path_to_pics='', already_cm_show_heat_map=already_cm_show_heat_map):
    '''
    My for now most anti-social line of code :
    len(ds.df_bdi.loc[pd.Index.intersection(ds.df_bdi.loc[ds.df_bdi['SCOREBDI']>19].index, [int(x) for x in ds.d_fcm.keys()])])
    Answering the question : how many of the patients with fcm have a bdi>19 
    '''
    print('Save directory will be : ', path_to_pics)
    for id_o in l_fcm.keys():
        id = int(id_o)
        if df_ds['SCOREBDI'].loc[id]<10 and df_ds['SCOREBDI'].loc[id]>=0:
            already_cm_show_heat_map(l_fcm[str(id)], os.path.join(path_to_pics,str('lowbdi/'+str(id)+'_'+str(int(df_ds['SCOREBDI'].loc[id]))+'_hm.png')))
            print('FCM added in lowbdi for :', id) 
        elif df_ds['SCOREBDI'].loc[id]>=10 and df_ds['SCOREBDI'].loc[id]<=19:
            already_cm_show_heat_map(l_fcm[str(id)], os.path.join(path_to_pics,str('midbdi/'+str(id)+'_'+str(int(df_ds['SCOREBDI'].loc[id]))+'_hm.png')))
            print('FCM added in midbdi for :', id)
        elif df_ds['SCOREBDI'].loc[id]>19 :
            already_cm_show_heat_map(l_fcm[str(id)], os.path.join(path_to_pics,str('highbdi/'+str(id)+'_'+str(int(df_ds['SCOREBDI'].loc[id]))+'_hm.png')))
            print('FCM added in high bdi for :', id)
        elif df_ds['SCOREBDI'].loc[id]<0 or df_ds['SCOREBDI'].loc[id]>=60:
            print('Unvalid value for ', id, ' - bdi = ', df_ds['SCOREBDI'].loc[id])
        else :
            print(id,' could not be associated a BDI value !')

def scatter_plot_features(df,name='test',f1="Age",f2="rh_supramarginal_volume", other=None):
    """
    Plots two features as a scatterplot
    """
    sns.set_style("whitegrid")
    temp_df=pd.concat([df[f1],df[f2],df[other]],axis=1)
    plot = sns.scatterplot(data=temp_df,x=f1,y=f2,hue=other).get_figure()
    plot.savefig(name+'/'+'sp_'+f2+'_'+f1+'.png')
    plt.clf()

def dist_plot_features(df,name='test',f1="Age",f2="rh_supramarginal_volume", 
                       other=None):
    """
    Plots one features as a distplot
    """
    if f1==other:
        return False
    sns.set_style("whitegrid")
    print(f1)
#    temp_df=pd.concat([df[f1],df[f2],df[other]],axis=1)
    temp_serie=pd.Series(df[f1].copy(), name=str(f1))
    plot = sns.distplot(temp_serie).get_figure()
    plot.savefig(name+'_'+'dp_'+str(f1)+'.png')
    plt.clf()

def box_plot_features(df,name='test',f1="Age",f2="rh_supramarginal_volume", 
                       other=None):
    """
    Plots one feature as a box_plot. Use 'other' to separate between groups
    """
    if f1==other:
        return False
    
    plt.title(('Repartition of '+str(f1)))
    sns.set_style("whitegrid")
    temp_df=pd.concat([df[f1],df[other]],axis=1)
    plot = sns.boxplot(data=temp_df,x=f1,hue=other,palette="Set3")
    plot = sns.boxplot(data=temp_df,x=f1,hue=other,palette="Set3").get_figure()
    plot.savefig(name+'_'+'sbp_'+str(f1)+'.png')
    plt.clf()

lp = lambda i: plt.plot([],color=sc.cmap(sc.norm(i)), ms=np.sqrt(size), mec="none",
                        label="Feature {:g}".format(i), ls="", marker="o")[0]

def box_plot_features_cd(df,name='test',f1="Age",f2="rh_supramarginal_volume", 
                       other=None):
    """
    Plots one feature as a box_plot. Use 'other' to separate between groups
    """
    if f1==other:
        return False
    fig = plt.figure()
    sns.set_style("whitegrid")
    df_dict=dict()
    for dif in df[other].unique():
        df_dict[dif]=df.loc[df[other]==dif]
    
    for counter, dif in enumerate(df[other].unique()):
        plt.subplot(2, 1, counter+1) 
        if counter ==0 :plt.title(('Repartition of the '+str(f1)+' among patients'))
        sns.boxplot(data=df_dict[dif],
                           x=f1,
                           color =sns.color_palette("Paired", 8)[counter]).get_figure()
        plt.legend(labels=[dif],loc=4)

    plt.tight_layout()    
    plt.savefig(name+'_'+'sbp_with_hue_'+str(f1)+'.png')
    plt.show()
    plt.clf()
    plt.close('all')

def a_sl_of_plot(df_o,f1='Age',directory='plot_test',name='plots', target = 'ds', 
                 plot_function=dist_plot_features,other=None):
    df=df_o.copy()
    if not os.path.exists(directory):
        os.makedirs(directory)
    name = directory +'/'+ name
    df.describe().to_csv(str(name+'describe.csv'))
    for feature in list(df.columns):
#        try :   plot_function(df,name=name, f1=f1,f2=feature,other='ds')
#        except TypeError : print('Error for : ', feature)
        
        plot_function(df,name=name, f1=feature,f2=feature,other=other)
        
