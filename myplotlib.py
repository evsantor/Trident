#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.mplot3d import axes3d    

import seaborn as sns
from copy import deepcopy

from datetime import datetime, timedelta


convertTime = lambda x:datetime.utcfromtimestamp(x/1000)

from scipy.cluster import hierarchy
from math import log

def get_columns():
    return ["ETIME" , "CO_CYCL", "CO_INPC", "CO_FBND", "CO_BBND", "CO_BSPC", "CO_RTIR", "CO_PUT0", "CO_PUT1", "CO_PUT2", "CO_PUT3", "CO_PUT4", "CO_PUT5", "CO_PUT6", "CO_PUT7","ME_REBW", "ME_WRBW", "ME_PEMP", "ME_PMSS", "ME_PHIT", "IO_UTIL","IO_REBW", "IO_WRBW", "IO_RIOP", "IO_WIOP", "TIMESTAMP"]



def initialize_data(bmk, path='~/SWAN_projects/Trident1/newData/HS06_Nov8', columns=[ "ETIME", "CO_FBND", "CO_BSPC", "CO_RTIR", "CO_BBND"], newname=[]):
    dict_ = {}
    columns_ = get_columns()
    for ((k,v), n) in zip(bmk, newname):
        df = pd.read_csv(path+'/%s.proc' %k, sep=';',skipinitialspace=True,index_col=False,skiprows=1,names=columns_)[columns] 
        dict_[k] = {'view_name':n, 'range':v, 'data':df}
      
    return dict_


def get_splitted_data(dict_):
    
    file_df = pd.DataFrame()
    dict_splt = {}
   
    for key, val in (dict_.items()):     #for all the files
        df = val['data'].copy()
        v  = val['range']
        
        for i in range(len(v)-1):        #for all the red parts
            d =  df[ (df.ETIME > v[i]) & (df.ETIME < v[i + 1])].loc[:, df.columns != 'ETIME']
            if df.columns[1] == 'CO_FBND': 
                d['1 - (FB + BS + BB)'] = 1 - (df.CO_FBND + df.CO_BSPC + df.CO_BBND)
            
            file_df = file_df.append(d)
            
        dict_splt[key] = deepcopy(dict_[key])    
        dict_splt[key]['data'] = file_df
        file_df = file_df.iloc[0:0]
    
    return dict_splt


def get_plots(dict_, dict_mem, splt, splt_mem):

    for d in dict_:
        get_timeseries(dict_[d], "CPU")
        get_pairplot(splt[d], "CPU")
        get_timeseries(dict_mem[d], "Memory")
        get_pairplot(splt_mem[d], "Memory")

        
def get_timeseries(dict_, level):
   
    pdf = dict_['data']
    tss = dict_['range']
    
    pdf["ETIME"] = pdf.ETIME.astype(float)
    ax = pdf.plot(x='ETIME', y=pdf.columns[1:], figsize=(16,8),title='~~~~~~~~~~' + dict_['view_name'] + '  '+ level+ '~~~~~~~~~~', grid=True, subplots=True, lw=2, fontsize=12)
    
    
    for j in xrange(0, len(tss), 2):
        for i in range(len(ax)):
            ax[i].set_xlabel("Time", fontsize=15)
            ax[i].axvspan(tss[j], tss[j+1], color='red', alpha=0.3, zorder=10)
            
    
def get_pairplot(dict_, level):
    pdf = dict_['data']
    plt_range = dict_['plt_range']
    
    g = sns.pairplot(pdf, plot_kws={'alpha': 0.1})
    g.set(xlim=(0,plt_range[0]))
    g.set(ylim=(0,plt_range[1]))
    g.set(xlabel='Time')
    g.map(plot_mean)
    g.fig.suptitle(level, y=1.02)
    
    
def plot_mean(xdata, ydata, **kwargs):
    if xdata is ydata:
        return
    
    x = xdata.mean()
    y = ydata.mean()
    plt.scatter(x, y, color='r', marker='+')

def get_describe(splt_datasets): 
    all_describes = []
    
    dict_dscrb = {}
    for dataset in splt_datasets:
        wl_dscrb = []
        for (key, val) in dataset.items():
            dict_dscrb[key] = deepcopy(dataset[key])
            dict_dscrb[key]['data'] = val['data'].copy().describe()
            wl_dscrb.append(dict_dscrb[key])
        all_describes.append(wl_dscrb)
    return all_describes
    

def get_scatterplot(dict_,  level):
    pdf = dict_['data']
    name = dict_['view_name']
    
    pdf.plot.scatter(x="ME_REBW", y="ME_WRBW", alpha=0.3, s=10)
    plt.scatter(pdf["ME_REBW"].mean(), pdf["ME_WRBW"].mean(), color='r')
    ax = plt.gca()
    ax.set_title(name + ' ' + level)
    
def get_stackplot(dict_, range_, cols,  legend_names, ylim_, stacked_):
    
    
    fig, ax = plt.subplots()
    ax.set_facecolor("white")


    pdf = dict_['data'][cols]
    pal = sns.color_palette("Set1")

    #plt.rcParams.linewidth = 5
    pdf.plot(title=dict_['view_name'], kind='area', stacked=stacked_, ax=ax, fontsize=15, figsize=(10, 5), xlim=range_, ylim=(0,ylim_), color=pal, linewidth=2)


    ax.title.set_size(15)
    ax.set_xlabel("Time", fontsize=15)


    leg = ax.legend(legend_names, frameon=True, prop={'size': 10}, loc=1)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    for l in ax.lines:
        l.set_alpha(0)
        
        
def setData(name,fe,bs,be):
    return (name,
            pd.DataFrame.from_dict(
                {"fe bound s0":[fe], "bad spec s0":[bs], "be bound s0":[be]}).rename(index={0:'mean'}))



def get_ZLinkage(X, method='average', metric='euclidean'):
    '''
    In order to get the linkage of a matrix prod(X,X.T) 
    where X is a string
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    '''
    
    y = X
    Z = hierarchy.linkage(y, method=method, metric=metric)
    return Z


def myllf(label_list=None):
    def llf(id):
        if label_list is None:
            return id
        try:
            return label_list[id]
        except:
            return id
    return llf

def get_Dendrogram(X, names, p=30,  figsize=(5, 7), method='average', metric='euclidean',**kwargs):
    'make dendrogram and plot it. return the dendrogram and the linkage matrix'
    
  
    Z = get_ZLinkage(X, method, metric)
    
    plt.rcParams['lines.linewidth'] = 2
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor('w')
    ax.grid(b=True, color='grey', linewidth=0.5)
    ax.set_ylim([0,200])
    dn = hierarchy.dendrogram(Z, ax=ax, get_leaves=True, distance_sort=True, 
                              leaf_label_func=myllf(names), color_threshold=.18,
                              orientation='right', above_threshold_color='y', **kwargs)
    
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
 
    return Z,dn


def split_lhcb(data_dict, range_, plt_range, view_name):
    dict_ = {}
    dict_noEtime = {}
    
    dict_["lhcb-gen-sim.pmpe16.Trident.1"] = deepcopy(data_dict["lhcb-gen-sim.pmpe16.Trident.1"])
    df = dict_["lhcb-gen-sim.pmpe16.Trident.1"]['data'].copy()
    dict_["lhcb-gen-sim.pmpe16.Trident.1"]['range'] = range_
    dict_["lhcb-gen-sim.pmpe16.Trident.1"]['data'] = df[(df.ETIME > range_[0]) & (df.ETIME < range_[1])]
    
    for ((k, _), r) in zip(dict_.items(), plt_range):
        dict_[k].update({'plt_range': r}) 

    dict_["lhcb-gen-sim.pmpe16.Trident.1"]['view_name'] = view_name 
    
    dict_noEtime = deepcopy(dict_)
    df =  dict_["lhcb-gen-sim.pmpe16.Trident.1"]['data'].copy()
    dict_noEtime["lhcb-gen-sim.pmpe16.Trident.1"]['data'] = df.loc[:, df.columns != 'ETIME']

    return dict_, dict_noEtime