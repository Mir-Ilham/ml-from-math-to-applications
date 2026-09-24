# Other ML Algorithms

## 1. Inference Models

### I. Naive Bayes

#### Introduction
**Naive Bayes** is a probabilistic classifier based on **Bayes' Theorem**, with the "naive" assumption that all features are conditionally independent given the class label. Despite this simplifying assumption, it performs well in practice, especially for text classification.

Bayes' Theorem: `P(class | features) = P(features | class) * P(class) / P(features)`

#### Algorithm
1. Compute prior probabilities `P(class)` for each class from the training data.
2. Compute likelihoods `P(feature_i | class)` for each feature, assuming conditional independence.
3. For a new sample, compute the posterior for each class: `P(class | x) ∝ P(class) * Π P(x_i | class)`.
4. Predict the class with the highest posterior probability.

Common variants:
- **GaussianNB**: assumes continuous features follow a normal distribution.
- **MultinomialNB**: suited for discrete count data (e.g., word counts).
- **BernoulliNB**: suited for binary/boolean features.

#### Hyperparameters
| Hyperparameter | Description |
|---|---|
| `var_smoothing` | (GaussianNB) Portion of the largest variance added to all variances for stability. |
| `alpha` | (Multinomial/Bernoulli) Laplace/Lidstone smoothing parameter to handle zero probabilities. |
| `fit_prior` | Whether to learn class prior probabilities from data or assume uniform priors. |
| `class_prior` | Manually specified prior probabilities of classes. |

#### Example
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = GaussianNB(var_smoothing=1e-9)
nb.fit(X_train, y_train)

print("Accuracy:", nb.score(X_test, y_test))
```

---

### II. K-Nearest Neighbors

#### Introduction
**K-Nearest Neighbors (KNN)** is a non-parametric, instance-based ("lazy") learning algorithm used for classification and regression. It predicts a query point's label based on the labels of its `k` closest points in the feature space.

#### Algorithm
1. Choose the number of neighbors `k` and a distance metric (e.g., Euclidean).
2. For a query point, compute the distance to every training point.
3. Select the `k` nearest training points.
4. Predict via majority vote among the neighbors' labels (classification) or the average of their values (regression).

KNN has no explicit training phase — all computation happens at prediction time.

#### Hyperparameters
| Hyperparameter | Description |
|---|---|
| `n_neighbors` | Number of neighbors (`k`) to consider. |
| `weights` | Weighting scheme (`uniform` or `distance`) for neighbor votes. |
| `metric` | Distance metric used (e.g., `euclidean`, `manhattan`, `minkowski`). |
| `p` | Power parameter for the Minkowski metric. |
| `algorithm` | Method used to compute nearest neighbors (`auto`, `ball_tree`, `kd_tree`, `brute`). |

#### Example
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2)
knn.fit(X_train, y_train)

print("Accuracy:", knn.score(X_test, y_test))
```

---

## 2. Kernel Based

### I. SVM

#### Introduction
**Support Vector Machine (SVM)** is a supervised learning algorithm that finds the optimal hyperplane separating classes by maximizing the margin between them. Using the **kernel trick**, SVMs can also model non-linear decision boundaries by implicitly mapping data into a higher-dimensional space.

#### Algorithm
1. Find the hyperplane `w · x + b = 0` that maximizes the margin `2 / ‖w‖` between the closest points of each class (the "support vectors").
2. For non-perfectly-separable data, introduce slack variables and a regularization parameter `C` to allow a soft margin (trading off margin width against misclassification).
3. For non-linear boundaries, apply a **kernel function** (linear, polynomial, RBF, sigmoid) that computes similarity in a higher-dimensional space without explicitly transforming the data.
4. Classify new points based on which side of the resulting decision boundary they fall on.

#### Hyperparameters
| Hyperparameter | Description |
|---|---|
| `C` | Regularization strength; trades off margin size against misclassification. |
| `kernel` | Kernel function used (`linear`, `poly`, `rbf`, `sigmoid`). |
| `gamma` | Kernel coefficient for `rbf`, `poly`, and `sigmoid`; controls influence of individual points. |
| `degree` | Degree of the polynomial kernel (`poly` only). |
| `epsilon` | (SVR only) Margin of tolerance where no penalty is given for errors. |

#### Example
```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(C=1.0, kernel="rbf", gamma="scale")
svm.fit(X_train, y_train)

print("Accuracy:", svm.score(X_test, y_test))
```

---

## 3. Unsupervised ML (Clustering)

### I. K-Means Clustering

#### Introduction
**K-Means** is a centroid-based unsupervised clustering algorithm that partitions data into `K` clusters by minimizing the within-cluster sum of squared distances (variance) between points and their assigned cluster centroid.

#### Algorithm
1. Initialize `K` centroids (randomly, or using the `k-means++` strategy for better initial spread).
2. **Assignment step**: assign each data point to its nearest centroid.
3. **Update step**: recompute each centroid as the mean of the points assigned to it.
4. Repeat steps 2–3 until centroids stabilize (convergence) or a maximum number of iterations is reached.

#### Hyperparameters
| Hyperparameter | Description |
|---|---|
| `n_clusters` | Number of clusters (`K`) to form. |
| `init` | Centroid initialization method (`k-means++` or `random`). |
| `n_init` | Number of times the algorithm runs with different centroid seeds. |
| `max_iter` | Maximum number of iterations for a single run. |
| `tol` | Tolerance for declaring convergence. |

#### Example
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300, random_state=42)
labels = kmeans.fit_predict(X)

print("Cluster centers:", kmeans.cluster_centers_)
```

---

### II. DBSCAN

#### Introduction
**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** groups together points that are closely packed in dense regions, marking points in low-density regions as noise/outliers. Unlike K-Means, it does not require the number of clusters to be specified beforehand and can find arbitrarily shaped clusters.

#### Algorithm
1. Define a neighborhood radius `eps` and a minimum number of points `min_samples`.
2. Classify each point as:
   - **Core point**: has at least `min_samples` points (including itself) within `eps`.
   - **Border point**: within `eps` of a core point but not itself a core point.
   - **Noise point**: neither a core nor a border point.
3. Connect core points that are within `eps` of each other into the same cluster (density-reachability), and assign border points to the cluster of their neighboring core point.
4. Points labeled as noise are not assigned to any cluster.

#### Hyperparameters
| Hyperparameter | Description |
|---|---|
| `eps` | Maximum distance between two points for one to be considered in the neighborhood of the other. |
| `min_samples` | Minimum number of points required to form a dense region (core point). |
| `metric` | Distance metric used to compute neighborhoods (e.g., `euclidean`). |

#### Example
```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

dbscan = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")
labels = dbscan.fit_predict(X)

print("Number of clusters (excluding noise):", len(set(labels)) - (1 if -1 in labels else 0))
```
