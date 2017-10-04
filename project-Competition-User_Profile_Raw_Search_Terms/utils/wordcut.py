#!/usr/bin/env python
# -*- coding: UTF-8
import csv
import pdb
import os
import sys
import time

import pandas as pd
import numpy as np
from tqdm import tqdm


import jieba
import jieba.analyse
import jieba.posseg
reload(sys)
sys.setdefaultencoding('utf8')
jieba.enable_parallel() #parallel:speed up



def convert_csv(infile, label = 'train', outdir = './output/'):
    f = open(infile)
    outfile = os.path.join(outdir, 'orig_' + label + '.csv')
    
    # example:
    #   22DD920316420BE2DF8D6EE651BA174B    1   1   4   柔和双沟    女生  中财网首页 财经    http://pan.baidu.com/s/1plpjtn9 周公解梦大全查询2345    
    #   ID  age gender  education   querylist

    csvfile = file(outfile, 'wb')# the path of the generated test file
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'age', 'Gender', 'Education', 'QueryList'])
    for line in f:
        line.strip()
        data = line.split("\t")
        writedata = [data[0], data[1], data[2], data[3]]
        data[-1]=data[-1][:-1] # 换行符
        querystr = ''
        for d in data[4:]:
           try:
                querystr += d.decode('GB18030').encode('utf8') + '\t'
           except:
               print data[0],querystr
        querystr = querystr[:-1]
        writedata.append(querystr)
        writer.writerow(writedata)
        
def convert_labels(infile, label = 'train', outdir = './output/'):
    data = pd.read_csv(infile)
    data.ID.to_csv(outdir + label + '_id.csv', index = False)
    # three labels
    data.age.to_csv(outdir + label + '_age.csv', index = False)
    data.Gender.to_csv(outdir + label + '_gen.csv', index = False)
    data.Education.to_csv(outdir + label + '_edu.csv', index = False)
    # text file
    data.QueryList.to_csv(outdir + label + '_query.csv', index = False)
    

def word_cut(infile, label = 'train', outdir = './output/'):
    start = time.time() 
    infile = outdir + label + '_query.csv'
    outfile = outdir + label + '_jieba.csv'
    outfile_pos = outdir + label + '_POS.csv'
    
    data = pd.read_csv(infile, header = None)
    POS = {}
    # can check here for introduction of pos (Part-of-speech)   http://verbs.colorado.edu/~xuen/teaching/ling5200/ppts/pos-tagging1.pdf
    # allowPOS = ['n', 'v', 'x', 'eng', 'm', 'a', 'j']   # n名词，v动词，x特殊, eng 英语，m 数量词
    # allowPOS = ['n', 'v', 'x', 'eng', 'm', 'a', 'j']   # n名词，v动词，x特殊, eng 英语，m 数量词
    allowPOS = ['n', 'v', 'j']   # n名词，v动词，x特殊, eng 英语，m 数量词
    posdata = pd.DataFrame(columns = allowPOS.append('total'), dtype = int)
    worddata = pd.DataFrame(columns = ["words"])
    for i in tqdm(range(0, data.size)):
        summary = {}
        line = []
        words_keep = ""
        words = jieba.posseg.cut(data.iloc[i].values[0])
        total = 0
        for word, flag in words:
            flag = flag.encode('utf8')
            POS[flag] = POS.get(flag, 0) + 1
            if (flag[0] in allowPOS):
                words_keep = words_keep + " " + word
                summary[flag[0]] = summary.get(flag[0], 0) + 1
            total += 1
        summary['total'] = total # total number of words
        posdata = posdata.append(summary, ignore_index=True)
        worddata = worddata.append({'words': words_keep}, ignore_index=True)
    
    posdata.to_csv(outfile_pos, encoding = 'utf-8')
    worddata.to_csv(outfile, encoding = 'utf-8')
    print POS
    
    end = time.time() 
    print "total time: %f s" % (end - start)   