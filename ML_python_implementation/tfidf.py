#!/usr/bin/env python
# -*- coding: UTF-8

# http://blog.csdn.net/baimafujinji/article/details/51476117
#  TF-IDF算法解析与Python实现
# TF-IDF的基本思想是：词语的重要性与它在文件中出现的次数成正比，但同时会随着它在语料库中出现的频率成反比下降。 
# 但无论如何，统计每个单词在文档中出现的次数是必要的操作。所以说，TF-IDF也是一种基于 bag-of-word 的方法。



import nltk
import math
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer

import pdb

text1 = "Python is a 2000 made-for-TV horror movie directed by Richard \
Clabaugh. The film features several cult favorite actors, including William \
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy, \
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the \
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean \
Whalen. The film concerns a genetically engineered snake, a python, that \
escapes and unleashes itself on a small town. It includes the classic final\
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles, \
 California and Malibu, California. Python was followed by two sequels: Python \
 II (2002) and Boa vs. Python (2004), both also made-for-TV films."

# text2 = "Python, from the Greek word (πύθων/πύθωνας), is a genus of \
# nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are \
# recognised.[2] A member of this genus, P. reticulatus, is among the longest \
# snakes known."

# 去掉(πύθων/πύθωνας)，否则程序代码编码不通过
text2 = "Python, from the Greek word , is a genus of \
nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are \
recognised.[2] A member of this genus, P. reticulatus, is among the longest \
snakes known."


text3 = "The Colt Python is a .357 Magnum caliber revolver formerly \
manufactured by Colt's Manufacturing Company of Hartford, Connecticut. \
It is sometimes referred to as a \"Combat Magnum\".[1] It was first introduced \
in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued \
Colt Python targeted the premium revolver market segment. Some firearm \
collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy \
Thompson, Renee Smeets and Martin Dougherty have described the Python as the \
finest production revolver ever made."



pattern = r'''(?x)          # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''
    
pattern = r'''(?x)               # set flag to allow verbose regexps
              ([A-Z]\.)+         # abbreviations, e.g. U.S.A.
              | \$?\d+(\.\d+)?%? # numbers, incl. currency and percentages
              | \w+([-']\w+)*    # words w/ optional internal hyphens/apostrophe
              | [+/\-@&*]        # special characters with meanings
            '''    
             
#pattern = r"(?x)([A-Z]\.)+|\$?\d+(\.\d+)?%?|\w+([-']\w+)*|[+/\-@&*]" 
pattern = r'\w+|[^\w\s]+\(\)'
text = nltk.regexp_tokenize(text1, pattern)  


# 自己编写分词程序
def get_tokens(text):
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    #remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    #no_punctuation = lowers.translate(remove_punctuation_map)
    no_punctuation = lowers.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

# 去掉停用词
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        new = item.encode('utf-8')
        #stemmed.append(stemmer.stem(item))
        stemmed.append(new)
    return stemmed
    

# 自己编写IF-IDF程序
def tf(word, count):
    return 1. * count[word] / sum(count.values())

def n_containing(word, count_list):
    return sum(1. for text in count_list if word in text)

def idf(word, count_list):
    return math.log(len(count_list) / (1. + n_containing(word, count_list)))

def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)
    
    
countlist = [text1, text2, text3]
# bloblist = [document1, document2, document3]
# for i, blob in enumerate(bloblist):
#     print("Top words in document {}".format(i + 1))
#     scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
#     sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     for word, score in sorted_words[:3]:
#         print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
        
for i, text in enumerate(countlist):
    # 分词
    tokens = get_tokens(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed = stem_tokens(filtered, stemmer)
    count = Counter(stemmed)
    # 统计
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, count, countlist) for word in count}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
  
pdb.set_trace()
  
        
# sklearn 自带程序实现
corpus = ['This is the first document.',
      'This is the second second document.',
      'And the third one.',
      'Is this the first document?',]
      
corpus = countlist
tmp = vectorizer = TfidfVectorizer(min_df=1)
a=vectorizer.fit_transform(corpus)
b=vectorizer.get_feature_names()        
# 最终的结果是一个 4×9 矩阵。每行表示一个文档，每列表示该文档中的每个词的评分。
c=vectorizer.fit_transform(corpus).toarray()
print c
        