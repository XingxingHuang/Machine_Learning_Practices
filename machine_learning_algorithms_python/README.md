### Machine Learning Models

* [KNN](./knn.py) Realize a K nearest neighbour algorithm to find out the favorite language in one place. Tried different k to see the result. [scikit-learn KNN](http://scikit-learn.org/stable/modules/neighbors.html) has many nearest neighbor models.

![](./fig/knn_figure_1.png width = 100)

* [Naive Bayes](./naive_bayes.py) Realize Naive Bayes model to classify spam messages. [scikit-learn BernoulliNB](http://scikit-learn.org/stable/modules/naive_bayes.html) implements the same Naive Bayes algorithm we implemented here. Paul Graham’s articles [“A Plan for Spam”](http://www.paulgraham.com/spam.html) and [“Better Bayesian Filtering”](http://www.paulgraham.com/better.html) (are interesting and) give more insight into the ideas behind building spam filters.

* [Simple Linear Regression](linear_regression_simple.py) we were investigating the relationship between a DataSciencester user’s number of friends and the amount of time he spent on the site each day. Think about Why choose least squares in the model? (Maximum Likelihood Estimation).

```
Assumption of LR: the regression errors are normally distributed with mean 0 and some (known) standard deviation σ
```

* [Multiple Linear Regression](linear_regression_multiple.py) Compared to the simple linear regression, we collected additional data: for each of your users, you know how many hours he works each day, and whether he has a PhD. We will use this additional data to improve the model. We will talking about bootstrap to estimate the uncertainties and also the regulizations. scikit-learn provides [linear_model module](http://scikit-learn.org/stable/modules/linear_model.html) including a `LinearRegression` model similar to ours, as well as `Ridge` regression, `Lasso` regression, and other types of regularization too. [Statsmodels](http://www.statsmodels.org/stable/index.html) is another Python module that contains (among other things) linear regression models.


```
Assumption of multiple LR: 
the columns of x are linearly independent.
the columns of x are all uncorrelated with the errors ε.
```

* [Logistic Regression]()
looked at the problem of trying to predict which users paid for premium accounts. From logistic regression to SVM? The logistric regreesion model returns a boundary, a hyperplane that splits the parameter space into two half-spaces corresponding, an alternative approach to classfication is just to find such best hyperplane. This is the idea of SVM.


* [Decision Trees](./decision_trees.py)
There aer a number of job candidates from the site, with varying degrees of success. The data set consiste of several (qualitative) attributes of each candidate, as well as whether that candidate inter‐ viewed well or poorly. We will use decission tree to identify which candidates will interview well. scikit-learn has many [Decision Tree models](http://scikit-learn.org/stable/modules/tree.html). It also has an [ensemble module](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) that includes a RandomForestClassifier as well as other ensemble methods. [wiki of decission tree](https://en.wikipedia.org/wiki/Decision_tree_learning).



```
Key concept:
Entropy, describes how much information in the features. 
Decision trees are very easy to understand and interpret, and the process by which they reach a prediction is completely transparent. 
Decision trees can easily handle a mix of numeric and categorical and can even classify data for which attributes are missing.
Decision trees can be divided into classification trees (which produce categorical outputs) and regression trees (which produce numeric outputs).

At the same time, finding an “optimal” decision tree for a set of training data is computationally a very hard problem (NP-hard).
```

* [Neural Networks](./neural_networks.py) Implement a two layer neural network to recognize a CAPTCHA numbers。

```
Key concept to begin neural networks:
   Perceptrons
   sigmoid
   feed-forward
   Backpropagation
```
 
* [k-means](./) The above is supervised learning with a set of labeled data. Here we will implement an unsupervised learning with k-means. In the datasets, you know the locations of all your local users. You will choose meetup locations that make it convenient for everyone to attend. You will also use the datasets to implement a buttom up clustering method. Both of these two methods are not effecient in big datasets. scikit-learn has an entire module [sklearn.cluster](http://scikit-learn.org/stable/modules/clustering.html) that contains several clus‐ tering algorithms including KMeans and the Ward hierarchical clustering (different criterion) algorithm. SciPy has two clustering models scipy.cluster.vq (which does k-means) and scipy.cluster.hierarchy (different). 

* [MapReduce]()


### Thinking in Diffrent Ways.

* why L1 regulization works?

L1 regulizations will help to reduce the overfitting. It makes the weight sparse. But why sparse means prevent overfitting. Sparse means most weights are about 0, so these features are less important. You model only depend on several "importance" features. This could be treat as a way of feature selection. 

Another questions: what about if we fail to include one feature in our model? This question can be the oposite way of thinking the weights is sparse. If we drop one feature, it's the same as set the weight of this feature as 0, so the weight of other featuers should be increased/ decreased depending on whether the feauture is positive or native corellated. In this way, our model can be underfitted. 


### Useful informations for Data Scientist

[./python\_data\_scientist](./python_data_scientist) contains basic usage of python to [get data](./python_data_scientist/getting_data.py) from html, json, xml, api, basic [data virualization](./python_data_scientist/visualizing_data.py) like scatter, hist, bar etc., [probability](./python_data_scientist/hypothesis_and_inference.py) and the concept of A/B test. Explore a [gradient decent](./python_data_scientist/gradient_descent.py) by yourself. 


---

Many codes are from the book [Data Science from Scratch](https://github.com/joelgrus/data-science-from-scratch). Please check the github for more informations. 