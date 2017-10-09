#!/usr/bin/env python
# -*- coding: UTF-8
from pylab import ion
import pandas as pd 
# import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
import pdb

ion()

infile = {
    'train': './output/orig_train.csv',
    'test': './output/orig_test.csv',
    'train_pos': './output/train_POS.csv',
    'train_id': './output/train_id.csv.csv',
    'train_gender': './output/train_gen.csv',
    'train_age': './output/train_age.csv',
    'train_education': './output/train_edu.csv',
    'test_pos': './output/test_POS.csv'
}


age = pd.read_csv(infile['train_age'], header = None)
edu = pd.read_csv(infile['train_education'], header = None)
gen = pd.read_csv(infile['train_gender'], header = None)

sns.