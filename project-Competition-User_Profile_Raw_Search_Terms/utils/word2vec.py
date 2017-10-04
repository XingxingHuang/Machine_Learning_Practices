# coding=utf-8
import pdb, os
#from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.model_selection import KFold, StratifiedKFold
from gensim.models import word2vec
import xgboost as xgb
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tqdm import tqdm


class w2v():
    def __init__(self,size=300):
        random_rate = 8240
        self.size=size
        self.svc= SVC(C=1, random_state=random_rate)
        self.LR = LogisticRegression(C=1.0, max_iter=100, class_weight='balanced', random_state=random_rate, n_jobs=-1)
        self.clf = LinearSVC(random_state=random_rate)

    def fit(self, X, Y, T, func = 'lr'):
        """
        train and predict
        """
        # print 'fitting..'
        # self.LR.fit(X, Y)
        # res = self.LR.predict(T)        
        if func == 'lr':
            model = self.LR
        elif func == 'svc':
            model = self.svc
        model.fit(X, Y)
        res = model.predict(T)     
        return model, res

    def validation(self,X,y,fold_n =5, test = None, kind = 'label', func = 'lr'):
        """

        使用k-fold进行验证
        """
        print 'validating...'
        skf = StratifiedKFold(n_splits=fold_n, random_state = 0)
        score=[]
        for train_idx, test_idx in skf.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            model, res = self.fit(X_train, y_train, X_test, func = func)
            cur = sum(y_test == res) * 1.0 / len(res)
            score.append(cur)
        # predict for the train data
        print kind,
        for s in score:
            print ' %.3f' %s,
        print '\t\t %.4f' %np.mean(score)
        # predict for the test data
        if test != None:
            y_test = model.predict(test[0])
            score_test = sum(y_test == test[1]) * 1.0 / len(y_test)
            print 'score of test data: %.4f' %(score_test)
        return np.mean(score)

    def train_w2v(self, sentences, label = 'model_w2v', outdir = './output'):
        """
        训练wv模型
        :param 
        :return:none
        word2vec 参数列表
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,  
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,  
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,  
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH):  
        """
        print 'Training w2v, size = ', self.size
        model = word2vec.Word2Vec(sentences, size=self.size, window=100, workers=4)  # 训练模型; 注意参数window 对结果有影响 一般5-100
        outfile = outdir + label + '_%i.model' %self.size # 保存的model
        print 'Traning completed, save: ', outfile
        model.save(outfile)
        
    def load_trainsform(self, text, outdir = './output'):
        """
        载入模型，并且生成wv向量
        :param X:读入的文档，list
        :return:np.array
        """
        print 'Loading word2vec model, size = ', self.size
        model = word2vec.Word2Vec.load(os.path.join(outdir, 'model_w2v_%i.model' %self.size))
        res=np.zeros((text.words.size, self.size))

        print 'Transforming to word vectors...'
        for i in tqdm(range(text.words.size)):
            line = text.words.iloc[i].decode('utf-8')
            count=0
            for j,term in enumerate(line.split()):
                try:#---try失败说明X中有单词不在model中，训练的时候model的模型是min_count的 忽略了一部分单词
                    count += 1
                    res[i] += np.array(model[term])
                except:
                    1 == 1
            if count != 0:
                res[i] = res[i]/float(count) # 求均值 # 可以用CNN训练，但是长度会不一致出现问题
        return res

