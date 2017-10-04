#!/usr/bin/env python
# -*- coding: UTF-8
'''
Use the following commands in terminal to run:
    python run.py
Tips:
(1) Please set the step[1,2,3] to True, so you can run the program step by step.
(2) In step2, you should choose one or all of for "orders" from ["train", 'getvec', 'test'] to run each step.
(3) In step3, you should choose one or all of for "orders" from ["train", 'prediction'] to run each step.

'''
import csv
import pdb
import time
import os

import pandas as pd
import numpy as np

indir = './input/'
outdir = './output/'

##############################################
########## step 1: word cut ##########
##############################################
step1 = False
if step1:
    from utils.wordcut import convert_csv, convert_labels, word_cut

    # convert training and testing data into csv format
    # output:  orig_train.csv, orig_test.csv
    infile = os.path.join(indir, 'user_tag_query.2W.TRAIN')
    convert_csv(infile, label = 'train')
    infile = os.path.join(indir, 'user_validation_1w_test.dat')
    convert_csv(infile, label = 'test')

    # extract the labels and search terms from the training and testing data
    # output: 
    #   train_age.csv  train_edu.csv  train_gen.csv  train_id.csv train_query.csv
    #   test_*.csv
    infile = os.path.join(outdir, 'orig_train.csv')
    convert_labels(infile, label = 'train')
    infile = os.path.join(outdir, 'orig_test.csv')
    convert_labels(infile, label = 'test')

    # word cut for training and test data using jieba
    # output:  
    #   train_jieba.csv   train_POS.csv 
    #   test_*.csv
    infile = os.path.join(outdir, 'train_query.csv')
    word_cut(infile, label = 'train')
    infile = os.path.join(outdir, 'test_query.csv')
    word_cut(infile, label = 'test')
    pdb.set_trace()

##############################################
##########      step 2: word2vec    ##########
##############################################
# words
trainfile = os.path.join(outdir, 'train_jieba.csv')
testfile = os.path.join(outdir, 'test_jieba.csv')
data_train = pd.read_csv(trainfile)
data_test = pd.read_csv(testfile)
# labels
label_file_gen = os.path.join(outdir, 'train_gen.csv')
label_file_age = os.path.join(outdir, 'train_age.csv')
label_file_edu = os.path.join(outdir, 'train_edu.csv')
label_gen = np.loadtxt(label_file_gen, dtype = int)
label_age = np.loadtxt(label_file_age, dtype = int)
label_edu = np.loadtxt(label_file_edu, dtype = int)
label_test_gen = np.loadtxt(label_file_gen.replace('train', 'test'), dtype = int)
label_test_age = np.loadtxt(label_file_age.replace('train', 'test'), dtype = int)
label_test_edu = np.loadtxt(label_file_edu.replace('train', 'test'), dtype = int)

# gender data
index_gen = np.nonzero(label_gen)[0]
data_gen = data_train.words[index_gen]  # text
label_gen = label_gen[index_gen]
# age data
index_age = np.nonzero(label_age)[0]
data_age = data_train.words[index_age]
label_age = label_age[index_age]
# education data
index_edu = np.nonzero(label_edu)[0]
data_edu = data_train.words[index_edu]
label_edu = label_edu[index_edu]


step2 = False
if step2:
    print '---------w2v----------'
    from utils import word2vec
    
    # output:
    #   train_w2v.npy
    #   test_w2v.npy
    # order = "train"     # train word embedding model
    # order = "getvec"    # get word vectors
    # order = "test"      # cross-validation using word vectors only
    orders = ["train", 'getvec', 'test']
    orders = ['test']

    obj = word2vec.w2v(300)
    if "train" in orders:
        obj.train_w2v(data_train.words.tolist())
    if "getvec" in orders :
        vec_train = obj.load_trainsform(data_train)
        vec_test = obj.load_trainsform(data_test)
        np.save(os.path.join(outdir, "train_w2v"), vec_train)
        np.save(os.path.join(outdir, "test_w2v"), vec_test)
    if 'test' in orders:
        # ------------------------------------------------------
        w2vtrain = np.load(os.path.join(outdir, 'train_w2v.npy'))
        w2vtest = np.load(os.path.join(outdir, 'test_w2v.npy'))
        tmp_gen = w2vtrain[index_gen]
        tmp_age = w2vtrain[index_age]
        tmp_edu = w2vtrain[index_edu]
        res1 = obj.validation(tmp_gen, label_gen, test = [w2vtest, label_test_gen], kind = 'gender', func = 'lr')
        res2 = obj.validation(tmp_age, label_age, test = [w2vtest, label_test_age], kind = 'age', func = 'lr')
        res3 = obj.validation(tmp_edu, label_edu, test = [w2vtest, label_test_edu], kind = 'education', func = 'lr')
        print 'avg is:', (res1+res2+res3)/3.0
    pdb.set_trace()
    
'''
label:  gender
0.580 0.581 0.581 0.581 0.581        0.5806
label:  age
0.402 0.402 0.402 0.402 0.402        0.4021
label:  education
0.413 0.413 0.413 0.413 0.413        0.4131
avg is: 0.465280032988

'''    


##############################################
##########     step 3: classify     ##########
##############################################
from utils import classify
import multiprocessing

step3 = True
if step3:
    print '--------- classify ----------'
    # word vectors
    w2vtrain = np.load(os.path.join(outdir, 'train_w2v.npy'))
    w2vtest = np.load(os.path.join(outdir, 'test_w2v.npy'))
    w2vtrain_gen = w2vtrain[index_gen]
    w2vtrain_age = w2vtrain[index_age]
    w2vtrain_edu = w2vtrain[index_edu]

    # orders = "train"     # train the two level stack model and run cross-validation
    # orders = "predict"   # prediction for the test data
    # orders  = ["train", "predict"]     
    # orders  = ["predict"]  
    orders  = ["train"]  
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
        gender = termob.predict(data_gen, label_gen, data_test.words, w2vtrain_gen, w2vtest, 'gender')
        age = termob.predict(data_age, label_age, data_test.words, w2vtrain_age, w2vtest, 'age')
        edu = termob.predict(data_edu, label_edu, data_test.words, w2vtrain_edu, w2vtest, 'edu')
        
        print 'saving predictions'
        ID = pd.read_csv('./output/test_id.csv', header = None, names = ['ID']).ID
        data = pd.DataFrame(np.array([ID.tolist(), age.tolist(), gender.tolist(), edu.tolist()]).T, columns="ID age gen edu".split())
        data.to_csv(os.path.join(outdir, 'submit.csv'))
        pdb.set_trace()
        
        print 'check the test result'
        for kind in "age gen edu".split():
            y_test = data[kind].astype(int)
            y = pd.read_csv(os.path.join(outdir, 'test_%s.csv' %kind), header = None, names = [kind])[kind].astype(int)       
            score_test = np.sum(y_test == y) * 1.0 / len(y_test)
            print 'Test score for %s: %.4f' %(kind, score_test)
        # Test score for age: 0.6324
        # Test score for gen: 0.8552
        # Test score for edu: 0.6374
        # average: 0.7083
        
'''
[ 0.81571151  0.81802391]
gender  score mean: 0.816867711095
[ 0.59196734  0.58759245]
edu  score mean: 0.589779894615
[ 0.56107492  0.56328276]
age  score mean: 0.562178839998
'''
end=time.time()
print 'total time is', end-start
pdb.set_trace()
    
    


