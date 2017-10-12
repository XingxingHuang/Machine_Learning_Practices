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

* [LSTM](./LSTM.py) There is a good blog [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) that explains LSTM starting from RNN and includes the formulas.

![](./fig/LSTM3-chain.png width = 100)

* [Latent Factor Model](./latent_factor_model.py) Recommendation method, which divide the user-item matrix to two matrix. Recommendation includes Feature-based (or content-based) approach and Collaborative filtering (CF). The Factorization Methods is another method. [Matrix Factorization](./matrix_factorization.py) is one kind of latent factor model. While user-based or item-based collaborative filtering methods are simple and intuitive, matrix factorization techniques are usually more effective because they allow us to discover the latent features underlying the interactions between users and items. Author: [Albert Au Yeung](http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/). 潜在因子模型的本质是认为每个user、每个item都有一些潜在的因子（或者认为是topic），比如user U_1 = (0.5, 0.1, 0.2)，movie I_1 = (0.5, 0.2, 0.3)，其中三个维度分别代表：对于动作类型电影、喜剧类电影、历史题材电影的响应值，可以看出，用户U_1的对于动作类电影的响应很高（说明他爱看动作类电影），而电影I_1在动作类电影的相应很高（说明这部影片含有很多动作元素）。用向量点乘（U_1^TI_1）得到的值就代表了用户U_1对于电影I_1的打分预期。这个线性点乘的过程很好的吻合了直觉，而潜在因子的思想也很好的反映了人类真实决策过程，所以，这个模型是目前做推荐最流行也是比较准确的模型。



### Advanced Topic

* [NLP](./NLP.py) This includes examples about National Language Processing techniques like word cloud, n-gram models, grammar models, Gibbs sampling, topic model (LDA). [Natural Language Toolkit](http://www.nltk.org/) is a popular (and pretty comprehensive) library of NLP tools for Python and has a book [Natural Language Processing with Python](http://www.nltk.org/book/). [gensim](https://radimrehurek.com/gensim/) is a Python library for topic modeling better than the model here.

* [TF-IDF](./tfidf.py)  tf–idf, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. The formula is simple and can be found from [http://www.tfidf.com/](http://www.tfidf.com/).

* [Network Analysis, graphic](./network_analysis.py) This part is one the hardes. Here talks about many kinds of centralities and also talke about PageRank, please refer the book [Data Science from Scratch](https://github.com/joelgrus/data-science-from-scratch) for more details. 

* [Recommender Systems](./recommender_systems.py) Here includes several common recommending systems like Popular based, User-Based Collaborative Filtering, Item-Based Collaborative Filtering. There are frameworks named Crab and Graphlab for recommendations. You could check [Netflix Prize](http://www.netflixprize.com/index.html) for the famous competition in 2009.

* [MapReduce](./mapreduce.py) MapReduce is a programming model for performing parallel processing on large data sets. 1. Use a mapper function to turn each item into zero or more key-value pairs. (Often this is called the map function, but there is already a Python function called map and we don’t need to confuse the two.) 2. Collect together all the pairs with identical keys. 3. Use a reducer function on each collection of grouped values to produce output values for the corresponding key. The most widely used MapReduce system is [Hadoop](http://hadoop.apache.org/), Amazon.com offers an [Elastic MapReduce](https://aws.amazon.com/cn/emr/) service that can programmatically cre‐ ate and destroy clusters. [mrjob](https://github.com/Yelp/mrjob) is a Python package for interfacing with Hadoop (or Elastic MapReduce). Hadoop jobs are typically high-latency, which makes them a poor choice for “real-time” analytics. There are various “real-time” tools built on top of Hadoop, but there are also several alternative frameworks that are growing in popularity. Two of the most popular are [Spark](http://spark.apache.org/) and [Storm](http://storm.apache.org/).




### Thinking in Diffrent Ways.

* why L1 regulization works?

L1 regulizations will help to reduce the overfitting. It makes the weight sparse. But why sparse means prevent overfitting. Sparse means most weights are about 0, so these features are less important. You model only depend on several "importance" features. This could be treat as a way of feature selection. 

Another questions: what about if we fail to include one feature in our model? This question can be the oposite way of thinking the weights is sparse. If we drop one feature, it's the same as set the weight of this feature as 0, so the weight of other featuers should be increased/ decreased depending on whether the feauture is positive or native corellated. In this way, our model can be underfitted. 


### Useful informations for Data Scientist

[./python\_data\_scientist](./python_data_scientist) contains basic usage of python to [get data](./python_data_scientist/getting_data.py) from html, json, xml, api, basic [data virualization](./python_data_scientist/visualizing_data.py) like scatter, hist, bar etc., [probability](./python_data_scientist/hypothesis_and_inference.py) and the concept of A/B test. Explore a [gradient decent](./python_data_scientist/gradient_descent.py) by yourself. 

Databases you should know.  If you’d like to download a relational database to play with, [SQLite](http://www.sqlite.org/) is fast and tiny, while [MySQL](https://www.mysql.com/) and [PostgreSQL](https://www.postgresql.org/) are larger and featureful. If you want to explore NoSQL, [MongoDB](https://www.mongodb.com/) is very simple to get started with, which can be both a blessing and somewhat of a curse. It also has pretty good documentation. In [MongoDB](https://resources.mongodb.com/getting-started-with-mongodb), you will find very good document and even videos to introduced what it is an how it works.


---

Many codes are from the book [Data Science from Scratch](https://github.com/joelgrus/data-science-from-scratch). Please check the github for more informations. 