#!/usr/bin/env python
# -*- coding: UTF-8
from pylab import ion
import pandas as pd 
import matplotlib.pyplot as plt
from pylab import *
import pdb

ion()

infile = {
    'train': './data/orig_train.csv',
    'test': './data/orig_test.csv',
    'train_pos': './data/train_POS.csv',
    'train_id': './data/train_id.csv.csv',
    'train_gender': './data/train_gen.csv',
    'train_age': './data/train_age.csv',
    'train_education': './data/train_edu.csv',
    'test_pos': './data/test_POS.csv'
}

train = pd.read_csv(infile['train'])
test = pd.read_csv(infile['test'])


## 检查train 和 test中属性数量分布是否相同，有何特点。
# count = 1
# for plotstr in ['age', 'Gender', 'Education']:
#     figure()
#     # plt.subplot(2, 2, count)
#     count += 1
#     plt.title = plotstr
#     train.groupby(plotstr).count().ID.plot()
#     test.groupby(plotstr).count().ID.plot()
#     plt.grid()
#     pdb.set_trace()

## 检查不同label 下面，不同磁性单词的分布随着label 不同是否有区别。
posdata = pd.read_csv(infile['train_pos'], encoding = 'utf-8')
for plotstr in ['age', 'Gender', 'Education']:
    
    plotdata = pd.read_csv(infile['train_%s' %plotstr.lower()], header = None)
    data = pd.concat([posdata, train], axis = 1)
    
    keys1 = 'a j m n v x total'.split() #posdata.keys().tolist()
    keys2 = 'a j m n v x'.split()
    keys = posdata.keys().tolist()+['age', 'Gender', 'Education']
    
    # plot 1, 每个label 不同属性的次数
    data[keys].groupby(plotstr).sum().divide(train.groupby(plotstr).count().ID, axis=0)[keys1].plot()  # 扣除了人数的影响，但是注意没有扣除每个人搜索词长度的影响
    plt.title = plotstr
    plt.grid()
    plt.ylim(10, 1000)
    
    # plot 2，每个label，不同属性的词频
    figure()
    posdata[keys1].divide(posdata['total'], axis = 0).median().plot() # 词频的分布, 不同种类
    plt.title = plotstr
    plt.grid()
    
    # plot 3，每个label，不同属性的词频
    data[keys2] = data[keys2].divide(data['total'], axis = 0)
    data['keep'] = data[keys2].sum(axis = 1)
    data.groupby(plotstr).median()[keys2 + ['keep']].plot() # 词频的分布， 不同的label
    plt.title = plotstr
    plt.grid()    
    
    print data.groupby(plotstr).median() # 词频的区别
    pdb.set_trace()
    
