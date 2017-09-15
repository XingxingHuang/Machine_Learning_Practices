# coding=utf-8
import pdb
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

    def fit(self, X, Y, T):
        """
        train and predict
        """
        # print 'fitting..'
        # self.LR.fit(X, Y)
        # res = self.LR.predict(T)
        self.svc.fit(X, Y)
        res = self.svc.predict(T)        
        return res

    def validation(self,X,y,fold_n =5, kind = 'label'):
        """

        使用k-fold进行验证
        """
        print 'validating...'
        skf = StratifiedKFold(n_splits=fold_n, random_state = 0)
        score=[]
        for train_idx, test_idx in tqdm(skf.split(X, y)):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            res = self.fit(X_train, y_train, X_test)
            cur = sum(y_test == res) * 1.0 / len(res)
            score.append(cur)
        print "label: ", kind
        print score, np.mean(score)
        return np.mean(score)

    def train_w2v(self, sentences, label = 'model_w2v', outdir = './data/'):
        """
        训练wv模型
        :param 
        :return:none
        """
        print '正在训练w2v 针对语料, size = ', self.size
        model = word2vec.Word2Vec(sentences, size=self.size, window=100, workers=4)  # 训练模型; 注意参数window 对结果有影响 一般5-100
        outfile = outdir + label + '_%i.model' %self.size # 保存的model
        print '训练完毕，已保存: ', outfile
        model.save(outfile)
        
    def load_trainsform(self, text):
        """
        载入模型，并且生成wv向量
        :param X:读入的文档，list
        :return:np.array
        """
        print '载入模型中, size = ', self.size
        model = word2vec.Word2Vec.load('./data/model_w2v_%i.model' %self.size) 
        res=np.zeros((text.words.size, self.size))

        print '生成w2v向量中.'
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

