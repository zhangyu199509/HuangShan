# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:05:39 2015

@author: hwx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import time

area_index = 'floor'
area_len =9
tw_span = 10.0   #定义时间窗大小（分钟数）
hour_max = 3    #定义分析的最长时间序列（小时数）

tw_cnt = 60/tw_span * hour_max

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

#用来规整matrix数据格式的准备工作及子程序（在载入数据时会用到）
tw=np.arange(60/tw_span * hour_max).astype(int)
areas = (np.arange(area_len)+1).astype(int)
miindex = pd.MultiIndex.from_product([areas, tw], names=['area', 'tw'])
area_tw = pd.DataFrame([0 for i in range(len(areas)*len(tw))], columns=['v'], index=miindex).reset_index()
area_tw['seq'] = (area_tw.area-1)*tw_cnt + area_tw.tw
del area_tw['v']

def merge_index(group):
    temp = pd.merge(group, area_tw, how='right', on=['tw', 'area'])
    return temp 


#根据时间窗来压缩trace数据
def time_window_compress(group):
    t_min = group.hour.min()
    group['tw'] = ((group.hour-t_min)*60/tw_span).astype(int)
    tw_area = pd.DataFrame({'area':group.groupby('tw').apply(process_mode).astype(int)})
    tw_area = tw_area.reset_index()
    tw_area['show'] = 1
    return (tw_area)
def process_mode(group):
    return (stats.mode(group[area_index])[0])
    
#载入数据，并按照AP点进行压缩    
daymax = 100
files = os.listdir('E:/FuturePlaza/weilai_AP') #此处需要修改数据文件路径
day = 0
for file in files:
    print file,
    #import data with AP label
    path = 'e:/FuturePlaza/weilai_AP/%s' %file  #此处需要修改数据文件路径
    frame = pd.read_csv(path, names=['floor', 'mac', 'day', 'hour', 'ap_seq'])
    del frame['floor']
    frame = pd.merge(frame, centroids, on=['ap_seq'])
    frame['floor'] = frame['floor']+3
    frame_grouped = frame.groupby(['day','mac'])
    #用于记录每日mac的dataframe，可以和matrix对应起来
    r1 = []
    
    #对原始数据进行压缩，计算每个tw中的最主要区域
    temp = []
    for mac, data in frame_grouped:
        trace_tw = time_window_compress(data)
        if ((trace_tw.tw.max()< 300/tw_span) & (trace_tw.tw.max()> 20/tw_span)):
            trace_tw = merge_index(trace_tw).fillna(0).sort('seq')
            temp.append(trace_tw.show)
            r1.append(pd.DataFrame({'day':[data.iloc[0].day], 'mac':[mac[1]]}))                
    temp = pd.concat(temp, ignore_index = True)  
    r1 = pd.concat(r1, ignore_index = True)
    m1 = np.reshape(temp.values.flatten(), (len(temp)/area_len/tw_cnt, area_len*tw_cnt))    
    if day == 0:
        matrix = m1
        rec = r1
    else:
        matrix = np.vstack((matrix, m1))
        rec = rec.append(r1)
    day = day + 1
    if day > daymax:
        break 



################################################
import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
n_row, n_col = 4, 9
n_components = n_row * n_col
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
image_shape = (area_len, tw_cnt)
rng = RandomState(0)

##################################################
n_samples, n_features = matrix.shape

# global centering
matrix_centered = matrix - matrix.mean(axis=0)

# local centering
#matrix_centered -= matrix_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d samples" % n_samples)

###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

## NMF
###############################################################################

name ='Non-negative components - NMF'
estimator = decomposition.NMF(n_components=n_components, init='nndsvda', beta=5.0, tol=5e-3, sparseness='components')
t0 = time()
data = matrix
estimator.fit(data)
train_time = (time() - t0)
print("done in %0.3fs" % train_time)

result = estimator.fit_transform(data)
print result.sum()

cln = pd.Series('v'+str(i) for i in range(n_components))
temp = pd.DataFrame(result, columns = cln)
nmf_fit = pd.merge(rec, temp, left_index=True, right_index=True).set_index(['day','mac'])

components_ = estimator.components_
plot_gallery('%s - Train time %.1fs' % (name, train_time), components_[:n_components])
plt.show()

rho = nmf_fit.corr()
pval = np.zeros(n_components)
for i in range(n_components):
    for j in range(n_components):
        JonI = pd.ols(y=nmf_fit.icol(i), x=nmf_fit.icol(j), intercept=True)
        pval[i,j]  = JonI.f_stat['p-value']

