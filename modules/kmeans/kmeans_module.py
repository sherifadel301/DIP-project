import numpy as np
import matplotlib.pyplot as plt

# =============================================
# 1) Initialize Centroids
# =============================================
def initialize_centroids(data, k):
    """
    Randomly select k unique data points as initial centroids
    """
    np.random.seed(42)
    random_indices = np.random.choice(len(data), size=k, replace=False)
    return data[random_indices]


# =============================================
# 2) Assign Each Point to the Nearest Centroid
# =============================================
def assign_clusters(data, centroids):
    """
    Calculate Euclidean distance between each point and each centroid
    Return an array of cluster labels
    """
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


# =============================================
# 3) Update Centroids
# =============================================
def update_centroids(data, labels, k):
    """
    Recalculate the centroid of each cluster as a mean of assigned points
    """
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


# =============================================
# 4) K-means Main Loop
# =============================================
def kmeans(data, k, max_iters=100):
    """
    Full K-means algorithm implementation
    """
    centroids = initialize_centroids(data, k)

    for i in range(max_iters):
        old_centroids = centroids.copy()

        # Step 1: Assign clusters
        labels = assign_clusters(data, centroids)

        # Step 2: Update centroids
        centroids = update_centroids(data, labels, k)

        # Check convergence (if centroids did not change)
        if np.allclose(centroids, old_centroids):
            print(f"Converged after {i+1} iterations.")
            break

    return centroids, labels


# =============================================
# 5) Plot Clusters
# =============================================
def plot_clusters(data, labels, centroids, title="K-means Clustering"):
    """
    Visualize clusters and centroids (2D only)
    """
    plt.figure(figsize=(8, 6))

    k = len(centroids)
    for i in range(k):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                marker='*', s=300, c='black', label="Centroids")

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================
# 6) Test Code (Optional)
# =============================================
if __name__ == "__main__":
    # Example test on random data
    from sklearn.datasets import make_blobs
    
    data, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    
    k = 3
    centroids, labels = kmeans(data, k)
    
    plot_clusters(data, labels, centroids)
