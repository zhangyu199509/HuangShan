# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:05:39 2015

@author: hwx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import os
import time

area_index = 'floor'
area_len =9

#目前数据的起始时间是'2015-03-26 00:00:00', Friday
day_start =pd.to_datetime('2015-03-26 00:00:00')

#import data
centroids = pd.read_csv('e:/FuturePlaza/centroids.csv')
centroids['ap_seq'] = centroids.idx + centroids.floor*100
#替换-3层5号ap的位置和楼层（从数据上看这个点应该是放错位置了）
centroids.loc[5,'x']= 150
centroids.loc[5,'y'] = -65
centroids.loc[5,'floor'] = 0 
del centroids['x']
del centroids['y']
del centroids['idx']

#载入经过整理的轨迹数据
trace_comp = pd.read_csv('e:/FuturePlaza/trace.csv',names=['floor','mac','day','hour','ap_seq','ap_next','ap_stay','cnt']) 
trace_comp.floor = trace_comp.floor + 3
#载入每条轨迹的统计信息
tr = pd.read_csv('e:/FuturePlaza/tr_property.csv') 
#载入每个mac地址的统计信息
mp = pd.read_csv('e:/FuturePlaza/mac_property.csv')

user = mp[mp.user==True]
stuff = mp[mp.user==False]

#对user的trace中的时间间隔，将其转换为log
tr_user = trace_comp[(trace_comp.mac.isin(user.mac)) & (trace_comp.ap_stay>0.01)]

#按照area对trace进行压缩
def area_compress(group):
    group = group.sort('hour')   #按时间排序，非常重要！
    group = group[(group[area_index] == group[area_index].shift(1)) & 
        (group.ap_stay<(1.0/2))  == False]    #剔除与前后点在同一位置的点，但保留1/2小时以上信号间隔的点（此处将gap的敏感尺度设定为比正常情况数量大一级）
    group = group.sort('hour')   #按时间排序，非常重要！
    group['gap'] = (group.ap_stay == group.hour-group.hour.shift(1))
    group['ap_stay'] = group.hour.shift(-1) - group.hour
    return (group)       
trace_area = tr_user.groupby(['mac','day']).apply(area_compress)
trace_area = trace_area.dropna()


#载入压缩好的trace_area数据
trace_area = pd.read_csv('e:/FuturePlaza/trace_area.csv')
trace_area['stay_log'] = (np.log(trace_area.ap_stay*64)/np.log(4)).astype(int)
#ts_len = trace_area.groupby(['mac','day']).stay_log.sum().max()
ts_len = 20 
#group = trace_area[(trace_area.day==43)&(trace_area.mac=='80414ebbbcb5')]
#生成一个log时长的Matrix
def log_window_compress(group):      
    step_start = 0
    s = np.zeros(ts_len * area_len)
    for i in range(len(group)):
        if group.iloc[i].gap:            
            step_end = step_start + group.iloc[i].stay_log-1    #此处将gap的敏感尺度设定为比正常情况数量大一级
            #不做任何处理，即gap的信号量为0
        else:
            step_end = step_start + group.iloc[i].stay_log
            f = group.iloc[i][area_index]*ts_len
            s[f+step_start : f+step_end] = 1            
        step_start = step_end
        df = pd.DataFrame({'v': pd.Series(s)})
    return(df)        
rec = trace_area.groupby(['mac','day']).apply(log_window_compress)
rec.to_csv('e:/FuturePlaza/rec4matrix_log.csv')

rec=pd.read_csv('e:/FuturePlaza/rec4matrix_log.csv')
matrix = np.reshape(rec.values.flatten(), (len(rec)/area_len/ts_len, area_len*ts_len)) 
rec = rec.reset_index()
del rec['v']


################################################
import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn import decomposition

##################################################
n_samples, n_features = matrix.shape
print("Dataset consists of %d samples" % n_samples)

###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        #vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=0, vmax=comp.max())
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

## NMF
###############################################################################
error = []
for i in range(7):
    n_components = 2**i   
    n_row, n_col = int(n_components/4)+1, 4

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    image_shape = (area_len, ts_len)
    rng = RandomState(0)
    
    name ='Non-negative components - NMF'
    estimator = decomposition.NMF(n_components=n_components, init='nndsvd', beta=5.0, tol=5e-3, sparseness='components')
    t0 = time()
    data = matrix
    estimator.fit(data)
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    error.append(np.sqrt(np.float(estimator.reconstruction_err_)/len(matrix)))
    
    cln = pd.Series('v'+ str(i) + str(j) for j in range(n_components))
    result = estimator.fit_transform(data)    
    temp = pd.DataFrame(result, columns = cln)
    if i==0:
        nmf_fit = temp
    else:
        nmf_fit = pd.merge(nmf_fit, temp, left_index=True, right_index=True)

    components_ = estimator.components_
    plot_gallery('%s - Train time %.1fs' % (name, train_time), components_[:n_components])
    plt.show()

'''
rho = nmf_fit.corr()
pval = np.zeros(n_components)
for i in range(n_components):
    for j in range(n_components):
        JonI = pd.ols(y=nmf_fit.icol(i), x=nmf_fit.icol(j), intercept=True)
        pval[i,j]  = JonI.f_stat['p-value']

