# K-means clustering
Unsupervised. Clustering algorithms try to find natural groupings in data. 
Similar data points (according to some notion of similarity) are considered in the same group. 
We call these groups clusters. There are K clusters. To find the optimal number of clusters, use the **elbow method**.

## How does it works?

* Place K **centroids** at random locations. Compute Euclidean distance of all points to the K centroids. 
* For each point, find the nearest centroid and assign the point to the respective cluster j.
* Now, we update the centroids. For each cluster, compute the mean of all points belonging to the cluster. The mean point 
is the new centroid.
* Repeat until convergence (when all centroids move less than a threshold alpha)

## Use cases

* Great when no labels available
* Numeric data. **Does not work with categorical data !!**, because we need to compute Euclidean distances. 
* Simple and fast
* Shine with multivariate data. Possible to run PCA (dimensionality reduction) then K-means clustering.

## How to chose K?

If you have an idea how many clusters there is, use that as K. If not, **elbow method** is used to chose optimal number of clusters K.

## Elbow method

* Run K-means for a range of values for K. Compute the sum of errors (SSE) : Euclidean distances between the points and their respective centroid. 
* Plot(number of clusters, sum of Euclidean distances). The plot should be a decreasing function, and should look like an elbow.
At some point, increasing the number of clusters do not decrease much the errors.
* Pick K that is right at the elbow. 

## Git examples

* [Fraud Detection](https://github.com/georgymh/ml-fraud-detection) 
* [MNIST digit classification without labels](https://github.com/Datamine/MNIST-K-Means-Clustering/blob/master/Kmeans.ipynb) 
* [PCA then K-means clustering](https://github.com/yesblogger/Data_Science/blob/master/K-means_from_scratch/K-means_clustering.ipynb)

## Credits

* Siraj Raval
* [gleydson](https://github.com/gleydson404)
