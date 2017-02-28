#!/usr/bin/env python
# -*- coding: UTF-8
# http://neuralnetworksanddeeplearning.com/chap1.html


# http://peekaboo-vision.blogspot.jp/2010/09/mnist-for-ever.html
# The dataset is very easy: random guessing is at 10% correct, a naive Bayes classifier scores about 90% correct and K nearest neighbor about 96.9 (I got that with K=3)

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import network


# take minutes to run
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# Epoch 29: 9488 / 10000
# 目前最高的可以达到 9979 / 10000

# take more to run  96.59%
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


# change learning rate
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)


# We might worry not only about the learning rate, but about every other aspect of our neural network. We might wonder if we've initialized the weights and biases in a way that makes it hard for the network to learn? Or maybe we don't have enough training data to get meaningful learning? Perhaps we haven't run for enough epochs? Or maybe it's impossible for a neural network with this architecture to learn to recognize handwritten digits? Maybe the learning rate is too low? Or, maybe, the learning rate is too high? When you're coming to a problem for the first time, you're not always sure.


# 如果仅仅采用灰度的方法，分类准确度可以达到  2225/10000。
# mnist_average_darkness.py


# 如果采用SVM，也可以得到比较好的分类结果，9435/10000。如果更仔细研究可以达到 98.5%
# 最好的SVM 结果参考这里  http://peekaboo-vision.blogspot.jp/2010/09/mnist-for-ever.html
# mnist_svm.py


# 采用交叉熵的方法  cross entropy 
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)