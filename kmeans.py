import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_dataset(name):
    return np.loadtxt(name)

dataset = load_dataset('durudataset.txt')


def plot(k, dataset, prototypes, belongs_to, iterat):


    for index in range(k):  # for index in range(k):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        plt.scatter(dataset[instances_close,0], dataset[instances_close,1])

    for i, prot in enumerate(prototypes):
        plt.scatter(prot[0], prot[1], color = 'black', marker = '*')

    plt.title('Iteration: {}'.format(iterat))
    plt.show()

class Kmeans:
    def __init__(self, k, dataset = dataset,  epsilon=0):
        self.dataset = dataset
        self.k = k
        self.epsilon = epsilon

    def euclidian(self, a, b):
        return np.linalg.norm(a - b)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for c in range(2):
            distance[labels.flatten() == c] = np.linalg.norm(X[labels.flatten() == c] - centroids[c], axis=1)
        return np.sum(np.square(distance))

    def fit(self):
        """
        :param k: The number of clusters (required)
        :param epsilon: The minimum error to be used in the stop condition (optional, default == 0)
        :param distance: The method is used to calculate the distance (Optional default == 0)
        :return: the centroids, the evolution history of centroids, the membership vector of each instance with its respective centroid
        """
        history_centroids = []
        # dataset = dataset[:, 0:dataset.shape[1] - 1]
        num_instances, num_features = self.dataset.shape
        prototypes = dataset[np.random.randint(0, num_instances - 1, size = self.k)]
        history_centroids.append(prototypes)
        prototypes_old = np.zeros(prototypes.shape)
        belongs_to = np.zeros((num_instances, 1))
        norm = self.euclidian(prototypes, prototypes_old)
        iteration = 0
        plot(self.k, self.dataset, prototypes, belongs_to, iteration)
        while norm > self.epsilon:
            iteration += 1
            norm = self.euclidian(prototypes, prototypes_old)
            prototypes_old = prototypes
            for index_instance, instance in enumerate(dataset):
                dist_vec = np.zeros((self.k, 1))
                for index_prototype, prototype in enumerate(prototypes):
                    dist_vec[index_prototype] = self.euclidian(prototype,instance)

                belongs_to[index_instance, 0] = np.argmin(dist_vec)

            tmp_prototypes = np.zeros((self.k, num_features))

            for index in range(len(prototypes)):
                instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
                prototype = np.mean(dataset[instances_close], axis=0)
                # prototype = dataset[np.random.randint(0, num_instances, size=1)[0]]
                tmp_prototypes[index, :] = prototype

            prototypes = tmp_prototypes

            history_centroids.append(tmp_prototypes)
            plot(self.k, self.dataset, prototypes, belongs_to, iteration)

        error = self.compute_sse(self.dataset, belongs_to, prototypes)

        return prototypes, history_centroids, belongs_to, error


Clusters = Kmeans(k=3)
prototypes, history_centroids, belongs_to, error = Clusters.fit()

# Elbow method

errors = []
for k_ in range(2,10):
    Clusters = Kmeans(k=k_)
    prototypes, history_centroids, belongs_to, error = Clusters.fit()
    errors.append(error)

plt.plot(list(range(2,10)), errors)
plt.title('Elbow method')
plt.show()
