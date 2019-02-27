'''
Reference:
    https://mubaris.com/posts/kmeans-clustering/
    https://github.com/mubaris/friendly-fortnight
'''

from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


def main():
    # Euclidean Distance Caculator
    def dist(a, b, ax=1):
        return np.linalg.norm(a - b, axis=ax)

    # Initial the centroids for clusters
    k = 3
    C_x = np.random.randint(0, np.max(X)-20, size=k)
    C_y = np.random.randint(0, np.max(X)-20, size=k)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    print("Initial Centroids")
    print(C)

    # Plotting along with the Centroids
    plt.scatter(f1, f2, c='#050505', s=7)
    plt.scatter(C_x, C_y, marker='*', s=200, c='g')

    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    clusters = np.zeros(len(X))
    error = dist(C, C_old, None)
    # Loop will run till the error becomes zero
    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)

    # check the results
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    return C


def test_sklearn():
    '''
        Test the algorithm in sklearn
    '''
    # Creating a sample dataset with 4 clusters
    X, y = make_blobs(n_samples=800, n_features=3, centers=4)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])

    # Initializing KMeans
    kmeans = KMeans(n_clusters=4)
    kmeans = kmeans.fit(X)
    labels = kmeans.predict(X)
    C = kmeans.cluster_centers_

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

if __name__ == '__main__':
    # Importing the dataset
    data = pd.read_csv('./data/kmeans_xclara.csv')
    print("Input Data and Shape")
    print(data.shape)
    data.head()

    # Getting the values and plotting it
    f1 = data['V1'].values
    f2 = data['V2'].values
    X = np.array(list(zip(f1, f2)))
    plt.scatter(f1, f2, c='black', s=7)

    # run the kmeans implementation
    C = main()

    # run the kemeans in scikit-learn
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans = kmeans.fit(X)
    labels = kmeans.predict(X)
    centroids = kmeans.cluster_centers_

    # Comparing with scikit-learn centroids
    print("Centroid values")
    print("Scratch")
    print(C) # From Scratch
    print("sklearn")
    print(centroids) # From sci-kit learn