import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random 
random_seed=42

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

class KMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.clusters = None
        self.random_seed = random_seed

    def fit(self, data, plot_progress=False):
        n_samples, n_features = data.shape
        if self.random_seed:
           np.random.seed(self.random_seed)
        self.centroids = data[np.random.choice(n_samples, self.k, replace=False)]

        for iteration in range(self.max_iterations):

            self.clusters = [[] for _ in range(self.k)]
            for point in data:
                distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
                closest_centroid = np.argmin(distances)
                self.clusters[closest_centroid].append(point)

            self.clusters = [np.array(cluster) for cluster in self.clusters]


            print(f"Iteration {iteration} distances:")
            for i, cluster in enumerate(self.clusters):
                if len(cluster) > 0:
                    distances = [euclidean_distance(point, self.centroids[i]) for point in cluster]
                    print(f"  Cluster {i + 1}: Average Distance = {np.mean(distances):.2f}, Max Distance = {np.max(distances):.2f}, Min Distance = {np.min(distances):.2f}")
                else:
                    print(f"  Cluster {i + 1}: No points assigned")


            if plot_progress:
                self.plot_iteration(data, iteration)


            new_centroids = np.array([
                cluster.mean(axis=0) if len(cluster) > 0 else self.centroids[i]
                for i, cluster in enumerate(self.clusters)
            ])




            if np.all(self.centroids == new_centroids):
                print(f"Convergence reached at iteration {iteration}.")
                self.centroids = new_centroids
                if plot_progress:
                    self.plot_iteration(data, iteration, final=True)
                break

            self.centroids = new_centroids

    def plot_iteration(self, data, iteration, final=False):
        colors = ["red", "blue", "green", "purple"]
        plt.figure(figsize=(8, 6))



        for i, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f"Cluster {i + 1}")



        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color="black", marker="x", s=200, label="Centroids")

        plt.title(f"K-Means Iteration {iteration}{' (Final)' if final else ''}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    def predict(self, data):
        predicted_clusters = []
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_centroid = np.argmin(distances)
            predicted_clusters.append(closest_centroid)
        return predicted_clusters

    def get_centroids(self):
        return self.centroids

    def get_clusters(self):
        return self.clusters



data, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)




plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], color="gray", label="Data Points", alpha=0.5)
plt.title("Initial Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()




kmeans = KMeans(k=4)


kmeans.fit(data, plot_progress=True)


centroids = kmeans.get_centroids()
clusters = kmeans.get_clusters()



def plot_final_clusters(data, centroids, clusters):
    colors = ["red", "blue", "green", "purple"]
    plt.figure(figsize=(8, 6))
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f"Cluster {i + 1}")
    plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="x", s=200, label="Centroids")
    plt.title("Final Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

plot_final_clusters(data, centroids, clusters)





def plot_specific_iteration(kmeans, iteration):
    """
    Plot the state of clusters and centroids at a specific iteration.
    """
    clusters = kmeans.clusters
    centroids = kmeans.centroids

    colors = ["red", "blue", "green", "purple"]
    plt.figure(figsize=(8, 6))
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], label=f"Cluster {i + 1}")
    plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="x", s=200, label="Centroids")
    plt.title(f"K-Means Iteration {iteration}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


specific_iteration = 3
plot_specific_iteration(kmeans, specific_iteration)