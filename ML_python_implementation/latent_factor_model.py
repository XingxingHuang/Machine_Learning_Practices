# coding:utf-8
# download from :
#   http://www.cnblogs.com/zhangchaoyang/articles/5517186.html
# LFM（Latent factor model， 相关定义，Content filtering， collaborator filtering。 Neighborhood methods
#   Learn hidden features in items and users that interact in such a way as to produce the ratings we've seen.
# SVD 也可以在相关网站中找到代码
# 隐语义模型LFM和LSI，LDA，Topic Model其实都属于隐含语义分析技术，是一类概念，他们在本质上是相通的，都是找出潜在的主题或分类。
# 基于矩阵分解(MF,Matrix Factorization)的推荐算法
# 相关问题：
#   冷启动问题，过拟合问题，对新物品/新用户的问题，时间复杂度问题
# 
# 概念对比：
#   Matrix Factorization
#   Fast matrix factorization (MF)
#   Factorization machines (FM)
#   SVD， SVD 是矩阵分解技术之一
#
# 这个博文包含相关的公式：
#   http://zhangyi.space/ji-yu-yin-yu-yi-mo-xing-latent-factor-modelde-dian-ying-tui-jian-xi-tong-ji-qi-zai-sparkshang-de-shi-xian/
#
# MF模型是推荐系统必备模型，参考yahoo的ppt
# http://www.ideal.ece.utexas.edu/seminar/LatentFactorModels.pdf

__author__ = "orisun"
 
import random
import math
 
 
class LFM(object):
 
    def __init__(self, rating_data, F, alpha=0.1, lmbd=0.1, max_iter=500):
        '''rating_data是list<(user,list<(position,rate)>)>类型
        '''
        self.F = F
        self.P = dict()  # R=PQ^T，代码中的Q相当于博客中Q的转置
        self.Q = dict()
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.rating_data = rating_data
 
        '''随机初始化矩阵P和Q'''
        for user, rates in self.rating_data:
            self.P[user] = [random.random() / math.sqrt(self.F)
                            for x in xrange(self.F)]
            for item, _ in rates:
                if item not in self.Q:
                    self.Q[item] = [random.random() / math.sqrt(self.F)
                                    for x in xrange(self.F)]
 
    def train(self):
        '''随机梯度下降法训练参数P和Q
        '''
        for step in xrange(self.max_iter):
            for user, rates in self.rating_data:
                for item, rui in rates:
                    hat_rui = self.predict(user, item)
                    err_ui = rui - hat_rui
                    for f in xrange(self.F):
                        self.P[user][f] += self.alpha * (err_ui * self.Q[item][f] - self.lmbd * self.P[user][f])
                        self.Q[item][f] += self.alpha * (err_ui * self.P[user][f] - self.lmbd * self.Q[item][f])
            self.alpha *= 0.9  # 每次迭代步长要逐步缩小
 
    def predict(self, user, item):
        '''预测用户user对物品item的评分
        '''
        return sum(self.P[user][f] * self.Q[item][f] for f in xrange(self.F))
 
if __name__ == '__main__':
    '''用户有A B C，物品有a b c d'''
    rating_data = list()
    rate_A = [('a', 1.0), ('b', 1.0)]
    rating_data.append(('A', rate_A))
    rate_B = [('b', 1.0), ('c', 1.0)]
    rating_data.append(('B', rate_B))
    rate_C = [('c', 1.0), ('d', 1.0)]
    rating_data.append(('C', rate_C))
 
    lfm = LFM(rating_data, 2)
    lfm.train()
    for item in ['a', 'b', 'c', 'd']:
        print item, lfm.predict('A', item)      #计算用户A对各个物品的喜好程度