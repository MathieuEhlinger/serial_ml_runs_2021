# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 21:57:02 2018

@author: Mathieu
"""

import sys
sys.path.append('./../datai')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datai.scorer import f1_scorer

#------------------------------
# Matplotlib graph generating functions

def pipeline_results_display(results, name, filename = None, save_d= True, 
                             scorer=f1_scorer):
    
    sns.set()
    plt.axes([0,0,1,1])
    plt.bar(list(results.keys()),list(results.values()), color='b')
    plt.xlabel('Type of score', fontsize=24)
    plt.ylabel('Score', fontsize=24)
    plt.legend
    plt.title(name+' True score')

    if save_d:
        assert filename!=None
        plt.savefig((filename+' tr.png'))
    plt.show()



def pipeline_retest_results_display(results, name, filename = None, 
                                    save_d= True,scorer=f1_scorer):
    
    sns.set()
    df=pd.DataFrame.from_dict(results)

    #plt.axes([0,0,15,1])
    
    plt.bar([str(x+1)+' '+scorer.__name__ for x in range(len(df.iloc[:, 0]))],df.iloc[:, 0], 
             color='b')
    plt.bar([str(x+1)+' Sensitivity' for x in range(len(df.iloc[:, 0]))],df.iloc[:, 1], 
             color='r')
    plt.bar([str(x+1)+' Specificity' 
             for x in range(len(df.iloc[:, 0]))],df.iloc[:, 2], color='c')
    
    
    plt.xlabel('Type of score', fontsize=24)
    plt.ylabel('Score', fontsize=24)
    plt.legend
    plt.title(name+' cross validation results')

    if save_d:
        assert filename!=None
        plt.savefig((filename+'Graph .png'))
    plt.xticks(rotation=-45)
    plt.show()
