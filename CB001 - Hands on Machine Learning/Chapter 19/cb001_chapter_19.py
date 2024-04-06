# -*- coding: utf-8 -*-
"""CB001 - Chapter 19.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/118Ub_6ERxbsUInT9041piWLWIIn9mjrc

#K-Means
"""

import numpy as np

# Generate random data points
np.random.seed(0)
# Number of clusters
k = 5
# Number of data points in each cluster
n = 300
# Generate random centroids
centroids = np.random.randn(k, 2) * 5
# Generate data points around the centroids
data = np.vstack([np.random.randn(n, 2) + centroid for centroid in centroids])
# Shuffle the data
np.random.shuffle(data)

# Visualize the data
import matplotlib.pyplot as plt

plt.scatter(data[:, 0], data[:, 1], s=10)
plt.title('Generated Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

from sklearn.cluster import KMeans

# Initialize KMeans with the desired number of clusters
kmeans = KMeans(n_clusters=k)

# Fit KMeans to the data
kmeans.fit(data)

# Get the cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualize the clustered data
plt.scatter(data[:, 0], data[:, 1], c=labels, s=10, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, color='red')
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

y_pred = kmeans.predict(data)
y_pred

kmeans.labels_

kmeans.cluster_centers_

from scipy.spatial import Voronoi, voronoi_plot_2d

# Fit Voronoi to the cluster centroids
vor = Voronoi(centroids)

# Plot the Voronoi diagram
voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2)

# Plot the clustered data and centroids
plt.scatter(data[:, 0], data[:, 1], c=labels, s=10, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, color='red')
plt.title('K-Means Clustering with Voronoi Tessellation')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Generate new random data points for testing
X_new = np.random.randn(100, 2) * 10

# Visualize the new data points
plt.scatter(X_new[:, 0], X_new[:, 1], s=10)
plt.title('New Data Points for Testing')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

kmeans.predict(X_new)

kmeans.transform(X_new)

"""# Centroid Initialization Methods

Knowing the actual positons of the centroids
"""

import numpy as np
from sklearn.cluster import KMeans

# Generate random data points
np.random.seed(0)
# Number of clusters
k = 5
# Number of data points in each cluster
n = 300
# Generate random centroids
centroids = np.random.randn(k, 2) * 5
# Generate data points around the centroids
data = np.vstack([np.random.randn(n, 2) + centroid for centroid in centroids])
# Shuffle the data
np.random.shuffle(data)

# Visualize the data
import matplotlib.pyplot as plt

plt.scatter(data[:, 0], data[:, 1], s=10)
plt.title('Generated Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Custom initialization
good_init = centroids

# Initialize KMeans with custom initialization
kmeans = KMeans(n_clusters=k, init=good_init, n_init=1)

# Fit KMeans to the data
kmeans.fit(data)

# Get the cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualize the clustered data
plt.scatter(data[:, 0], data[:, 1], c=labels, s=10, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, color='red')
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

kmeans.inertia_

kmeans.score(data)

"""#Non-random initialization and accelerated K-Means are used by default.

#Random initialization method
"""

# Initialize KMeans with custom initialization
kmeans2 = KMeans(n_clusters=k, init="random", n_init=1)

# Fit KMeans to the data
kmeans2.fit(data)

# Get the cluster centroids and labels
centroids2 = kmeans2.cluster_centers_
labels2 = kmeans2.labels_

# Visualize the clustered data
plt.scatter(data[:, 0], data[:, 1], c=labels2, s=10, cmap='viridis')
plt.scatter(centroids2[:, 0], centroids2[:, 1], marker='x', s=100, color='red')
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

"""#Full K-Means"""

# Initialize KMeans with custom initialization
kmeans3 = KMeans(n_clusters=k, init=good_init, n_init=1, algorithm="full")

# Fit KMeans to the data
kmeans3.fit(data)

# Get the cluster centroids and labels
centroids3 = kmeans3.cluster_centers_
labels3 = kmeans3.labels_

# Visualize the clustered data
plt.scatter(data[:, 0], data[:, 1], c=labels3, s=10, cmap='viridis')
plt.scatter(centroids3[:, 0], centroids3[:, 1], marker='x', s=100, color='red')
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

"""#Mini-Batch K-Means"""

from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=k)
minibatch_kmeans.fit(data)

# Get the cluster centroids and labels
centroids4 = minibatch_kmeans.cluster_centers_
labels4 = minibatch_kmeans.labels_

# Visualize the clustered data
plt.scatter(data[:, 0], data[:, 1], c=labels4, s=10, cmap='viridis')
plt.scatter(centroids4[:, 0], centroids4[:, 1], marker='x', s=100, color='red')
plt.title('Mini-Batch K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

"""#Finding the Optimal Number of Clusters"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate random data points
np.random.seed(0)
# Number of clusters
k_values = range(1, 11)
inertia_values = []

for k in k_values:
    # Fit KMeans to the data
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
    kmeans.fit(data)

    # Get the inertia value (within-cluster sum of squares)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.show()

"""With Silhouette Score:"""

from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate random data points
np.random.seed(0)

# Range of k values
k_values = range(2, 11)
silhouette_scores = []

# Generate data points
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
    kmeans.fit(data)

    # Calculate silhouette score
    silhouette_scores.append(silhouette_score(data, kmeans.labels_))

# Plot k vs Silhouette score
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.show()

from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.cm as cm

# Generate random data points
np.random.seed(0)
# Number of data points in each cluster
n = 300

# Range of k values
k_values = range(2, 12)

# Initialize subplot
fig, ax = plt.subplots(len(k_values), 1, figsize=(8, len(k_values) * 4))

for idx, k in enumerate(k_values):
    # Fit KMeans to the data
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
    kmeans.fit(data)

    # Calculate silhouette scores and sample silhouette values
    silhouette_scores = silhouette_score(data, kmeans.labels_)
    silhouette_values = silhouette_samples(data, kmeans.labels_)

    # Plot silhouette diagram
    y_lower = 10

    for i in range(k):
        cluster_silhouette_values = silhouette_values[kmeans.labels_ == i]
        cluster_silhouette_values.sort()
        cluster_size = cluster_silhouette_values.shape[0]
        y_upper = y_lower + cluster_size

        color = cm.nipy_spectral(float(i) / k)
        ax[idx].fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
        ax[idx].text(-0.05, y_lower + 0.5 * cluster_size, str(i))
        y_lower = y_upper + 10

    ax[idx].set_title(f"Silhouette plot for k={k}, silhouette score={silhouette_scores:.2f}")
    ax[idx].set_xlabel("Silhouette coefficient values")
    ax[idx].set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax[idx].axvline(x=silhouette_scores, color="red", linestyle="--")

    ax[idx].set_yticks([])
    ax[idx].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.tight_layout()
plt.show()

"""#Using clustering for image segmentation"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import data

# Load the camera image
image = data.camera()

# Normalize pixel values to range [0, 1]
image = np.array(image, dtype=np.float64) / 255

# Reshape the image to a 2D array of pixels
w, h = original_shape = tuple(image.shape)
image_array = np.reshape(image, (w * h, 1))

# Perform K-Means clustering on the pixel values
n_colors = 16  # Number of clusters (colors)
kmeans = KMeans(n_clusters=n_colors, random_state=0)
kmeans.fit(image_array)

# Predict cluster assignments for each pixel
labels = kmeans.predict(image_array)

# Replace each pixel with the mean of its assigned cluster
quantized_image = np.reshape(kmeans.cluster_centers_[labels], (w, h))

# Display original and segmented images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(kmeans.cluster_centers_.reshape(-1, 1), cmap='gray')
plt.title("Cluster Centers")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(quantized_image, cmap='gray')
plt.title("Quantized Image ({} colors)".format(n_colors))
plt.axis("off")

plt.show()

"""#Using Clustering for Preprocessing"""

from sklearn.datasets import load_digits
X_digits, y_digits = load_digits(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

log_reg.score(X_test, y_test)

"""Let's use K-Means for pre-processing the data"""

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
 ("kmeans", KMeans(n_clusters=50)),
 ("log_reg", LogisticRegression()),
 ])
pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test)

"""Let's use GridCV for knowing the appropriate number of clusters:"""

from sklearn.model_selection import GridSearchCV
param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

grid_clf.best_params_

grid_clf.score(X_test, y_test)

"""#Using Clustering for Semi-Supervised Learning"""

from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the dataset into labeled and unlabeled data
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.8, random_state=42)

# Create a semi-supervised learning setup
n_labeled_examples = X_labeled.shape[0]
n_clusters = len(np.unique(y_labeled))

# Fit K-Means on labeled data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_labeled)

# Assign pseudo-labels to unlabeled data based on K-Means clustering
pseudo_labels = kmeans.predict(X_unlabeled)

# Combine labeled and pseudo-labeled data
X_combined = np.vstack((X_labeled, X_unlabeled))
y_combined = np.hstack((y_labeled, pseudo_labels))

# Train a classifier on the combined labeled and pseudo-labeled data
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_combined, y_combined)

# Evaluate the classifier
accuracy = classifier.score(X, y)
print("Accuracy:", accuracy)