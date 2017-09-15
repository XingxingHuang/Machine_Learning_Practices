#!/usr/bin/env python
# -*- coding: UTF-8

import pandas as pd
import numpy as np
import csv
import pdb, os, sys
import time
reload(sys)


import jieba
import jieba.analyse
import jieba.posseg

from tqdm import tqdm


def convert_csv(infile, label = 'train', outdir = './data'):
    f = open(infile)
    outfile = os.path.join(outdir, 'orig_' + label + '.csv')
    
    # example:
    #   22DD920316420BE2DF8D6EE651BA174B    1   1   4   柔和双沟    女生  中财网首页 财经    http://pan.baidu.com/s/1plpjtn9 周公解梦大全查询2345    
    #   ID  age gender  education   querylist
    
    # fout = open(outfile, 'w')
    # for line in f.readlines():
    #     line = line.decode('GB18030').encode('utf8')
    #     # pdb.set_trace()
    #     line_new = line.strip().replace('\t', ',')
    #     fout.write(line_new + '\n')
    
    ##
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
        
def convert_labels(infile, label = 'train', outdir = './data/'):
    data = pd.read_csv(infile)
    data.ID.to_csv(outdir + label + '_id.csv', index = False)
    # three labels
    data.age.to_csv(outdir + label + '_age.csv', index = False)
    data.Gender.to_csv(outdir + label + '_gen.csv', index = False)
    data.Education.to_csv(outdir + label + '_edu.csv', index = False)
    # text file
    data.QueryList.to_csv(outdir + label + '_query.csv', index = False)
    

def word_cut(infile, label = 'train', outdir = './data/'):
    start = time.time() 
    infile = outdir + label + '_query.csv'
    outfile = outdir + label + '_jieba.csv'
    outfile_pos = outdir + label + '_POS.csv'
    #csvfile = open(outfile, 'w')
    
    data = pd.read_csv(infile, header = None)
    POS = {}
    # can check here for introduction of pos (Part-of-speech)   http://verbs.colorado.edu/~xuen/teaching/ling5200/ppts/pos-tagging1.pdf
    # allowPOS = ['n', 'v', 'x', 'eng', 'm', 'a', 'j']   # n名词，v动词，x特殊, eng 英语，m 数量词
    #allowPOS = ['n', 'v', 'x', 'eng', 'm', 'a', 'j']   # n名词，v动词，x特殊, eng 英语，m 数量词
    allowPOS = ['n', 'v', 'j']   # n名词，v动词，x特殊, eng 英语，m 数量词
    posdata = pd.DataFrame(columns = allowPOS.append('total'), dtype = int)
    worddata = pd.DataFrame(columns = ["words"])
    for i in tqdm(range(0, data.size)):
    # for i in tqdm(range(0, 100)):
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
    
    # for i in tqdm(range(0, data.size)):
    #     summary = {}
    #     line = []
    #     words_keep = ""
    #     words = jieba.posseg.cut(data.iloc[i].values[0])
    #     for word, flag in words:
    #         POS[flag] = POS.get(flag, 0) + 1
    #         if (flag[0] in allowPOS) and len(word)>=2:
    #             words_keep += word + " "
    #         summary[flag] = summary.get(flag, 0) + 1
    #     posdata.append(summary, ignore_index=True)
    #     line.append(words_keep.encode('utf8'))
    #     csvfile.write(" ".join(line)+'\n')
    # posdata.to_csv(outfile_pos)
    # csvfile.close()
    # print POS
    
    end = time.time() 
    print "total time: %f s" % (end - start)    
    # pdb.set_trace()



##### step 1: prepare csv file
# infile = './data/user_tag_query.2W.TRAIN'
# convert_csv(infile, label = 'train')

# infile = './data/user_validation_1w.dat'
# convert_csv(infile, label = 'test')

# infile = './data/orig_train.csv'
# convert_labels(infile, label = 'train')

# infile = './data/orig_test.csv'
# convert_labels(infile, label = 'test')


# trainname = './data/orig_train.csv'
# testname = './data/orig_test.csv'
# data = pd.read_csv(trainname)
# print data.info()
# data = pd.read_csv(testname)
# print data.info()


##### step 2: word cut

# sys.setdefaultencoding('utf8')
# #parallel:speed up
# jieba.enable_parallel()

# infile = './data/train_query.csv'
# word_cut(infile, label = 'train')
# infile = './data/test_query.csv'
# word_cut(infile, label = 'test')


####### prepare the data
# trainfile = './data/train_jieba.csv'
# testfile = './data/test_jieba.csv'
# data_train = pd.read_csv(trainfile)
# data_test = pd.read_csv(testfile)

# #载入label文件
# label_file_gen = './data/train_gen.csv'
# label_file_age = './data/train_age.csv'
# label_file_edu = './data/train_edu.csv'
# label_gen = np.loadtxt(label_file_gen, dtype = int)
# label_age = np.loadtxt(label_file_age, dtype = int)
# label_edu = np.loadtxt(label_file_edu, dtype = int)


# #以下为测试wv向量，仅仅使用wv向量做预测，目的在于寻找最好参数的WV向量
# print '载入所有的w2v向量中..'
# w2vtrain = np.load('./data/train_w2v.npy')
# #防止出现非法值
# if np.any((np.isnan(w2vtrain))):
#     print 'nan to num!'
#     pdb.set_trace()
#     w2vtrain = np.nan_to_num(w2vtrain)
# print '预处理中, 取出label 为0的元素'
# index = np.nonzero(label_gen)
# data_gen, label_gen = w2vtrain[index], label_gen[index]
# index = np.nonzero(label_age)
# data_age, label_age = w2vtrain[index], label_age[index]
# index = np.nonzero(label_edu)
# data_edu, label_edu = w2vtrain[index], label_edu[index]


##### step 3: word2vec
print '---------w2v----------'
from utils import word2vec

## order = "train"     # traing
# order = "getvec"  # get vectors
# order = "test"    # cross-validation
#orders = ["train", 'getvec', 'test']
# orders = ['test']
orders = []

obj = word2vec.w2v(300)
if "train" in orders:
    obj.train_w2v(data_train.words.tolist())
elif "getvec" in orders :
    vec_train = obj.load_trainsform(data_train)
    vec_test = obj.load_trainsform(data_test)
    np.save("./data/train_w2v", vec_train)
    np.save("./data/test_w2v", vec_test)
elif 'test' in orders:
    # ------------------------------------------------------
    res1 = obj.validation(data_gen, label_gen, kind = 'gender')
    res2 = obj.validation(data_age, label_age, kind = 'age')
    res3 = obj.validation(data_edu, label_edu, kind = 'education')
    print 'avg is:', (res1+res2+res3)/3.0
    
'''
label:  gender
[0.58758937691521962, 0.59335887611749683, 0.58007662835249041, 0.57598978288633462, 0.59463601532567045] 0.586330135919
label:  age
[0.34902060544390739, 0.36259541984732824, 0.36294222448460167, 0.3543788187372709, 0.37535014005602241] 0.360857441714
label:  education
[0.37286265857694428, 0.38279095421952564, 0.36948123620309054, 0.38189845474613687, 0.36802871341800109] 0.375012403433
avg is: 0.440733327022


label:  gender
[0.5842696629213483, 0.58722860791826315, 0.58467432950191567, 0.58544061302681993, 0.58518518518518514] 0.585359679711
validating...
5it [12:00, 143.94s/it]
label:  age
[0.4019333502925464, 0.4020356234096692, 0.40213794858742685, 0.40224032586558045, 0.40234275528393176] 0.402138000688
validating...
5it [10:26, 125.83s/it]
label:  education
[0.41312741312741313, 0.41312741312741313, 0.41307947019867547, 0.41307947019867547, 0.41330756488128106] 0.413144266307
avg is: 0.466880648902

'''    

##### prepare the data
trainfile = './data/train_jieba.csv'
testfile = './data/test_jieba.csv'
data_train = pd.read_csv(trainfile).words
data_test = pd.read_csv(testfile).words

w2vtrain = np.load('./data/train_w2v.npy')
w2vtest = np.load('./data/test_w2v.npy')

#载入label文件
label_file_gen = './data/train_gen.csv'
label_file_age = './data/train_age.csv'
label_file_edu = './data/train_edu.csv'
label_gen = np.loadtxt(label_file_gen, dtype = int)
label_age = np.loadtxt(label_file_age, dtype = int)
label_edu = np.loadtxt(label_file_edu, dtype = int)

#
index = np.nonzero(label_gen)[0]
data_gen = data_train[index].tolist()  # text
w2vtrain_gen = w2vtrain[index]
label_gen = label_gen[index]
#
index = np.nonzero(label_age)[0]
data_age = data_train[index].tolist()
w2vtrain_age = w2vtrain[index]
label_age = label_age[index]
#
index = np.nonzero(label_edu)[0]
data_edu = data_train[index].tolist()
w2vtrain_edu = w2vtrain[index]
label_edu = label_edu[index]


##### step 4: classify
from utils import classify
import multiprocessing

print '--------- classify ----------'
orders  = ['train']

start = time.time()
if "train" in orders:
    termob1 = classify.term()
    termob2 = classify.term()
    termob3 = classify.term()
    p1=multiprocessing.Process(target=termob1.validation,args=(data_gen, label_gen, w2vtrain_gen, 'gender',))
    p2=multiprocessing.Process(target=termob2.validation,args=(data_age, label_age, w2vtrain_age, 'age',))
    p3=multiprocessing.Process(target=termob3.validation,args=(data_edu, label_edu, w2vtrain_edu, 'edu',))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
    pdb.set_trace()
    
elif "predict" in orders:
    termob = classify.term()
    gender = termob.predict(data_gen, label_gen, data_test, w2vtrain_gen, w2vtest, 'gender')
    age = termob.predict(data_age, label_age, data_test, w2vtrain_age, w2vtest, 'age')
    edu = termob.predict(data_age, label_age, data_test, w2vtrain_age, w2vtest, 'edu')
    # ID = pd.read_csv('user_tag_query.10W.TEST.csv').ID
    # output('submit.csv', ID, age, gender, edu)
    pdb.set_trace()


end=time.time()
print 'total time is', end-start
    
pdb.set_trace()
    
    


