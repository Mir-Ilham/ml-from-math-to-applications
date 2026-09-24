# Tree-Based and Ensemble Learning Methods in Machine Learning

## Decision Trees

### Introduction
A **Decision Tree** is a supervised learning algorithm used for both classification and regression. It represents decisions as a tree structure, where each internal node tests a feature, each branch represents the outcome of the test, and each leaf node holds a prediction.

Key terms:
- **Root node**: the topmost node, representing the entire dataset.
- **Splitting**: dividing a node into sub-nodes based on a feature test.
- **Leaf node**: terminal node holding the final prediction.
- **Impurity**: a measure of how mixed the classes/values are within a node.

### Algorithm
Decision trees are built using **recursive binary splitting**:
1. At each node, evaluate all possible feature/threshold splits.
2. Choose the split that minimizes impurity (or maximizes information gain) in the resulting child nodes.
3. Recurse on each child node until a stopping condition is met (max depth, minimum samples, or a pure node).

Common impurity measures:
- **Gini Impurity** (classification): `Gini = 1 - Σ p_i²`
- **Entropy / Information Gain** (classification): `Entropy = -Σ p_i log2(p_i)`
- **Variance / MSE reduction** (regression)

**Pruning** (pre- or post-, e.g. cost-complexity pruning) is used to reduce overfitting by limiting tree size.

### Hyperparameters
| Hyperparameter | Description |
|---|---|
| `max_depth` | Maximum depth of the tree; controls overfitting. |
| `min_samples_split` | Minimum samples required to split an internal node. |
| `min_samples_leaf` | Minimum samples required at a leaf node. |
| `criterion` | Splitting quality measure (`gini`, `entropy`, `squared_error`, etc.). |
| `max_features` | Number of features considered when looking for the best split. |
| `ccp_alpha` | Complexity parameter for cost-complexity (post-)pruning. |

### Example
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42)
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))
```

---

## Introduction to Ensemble Learning

### Introduction
**Ensemble Learning** combines predictions from multiple models ("base" or "weak" learners) to produce a single, stronger model. The core idea is that a diverse group of models, when aggregated, generalizes better than any single model alone.

### Types of Ensembles
- **Bagging (Bootstrap Aggregating)**: trains multiple models independently on bootstrapped subsets of the data and aggregates predictions (voting/averaging). Primarily reduces **variance**. Example: Random Forest.
- **Boosting**: trains models sequentially, where each new model corrects the errors of its predecessors. Primarily reduces **bias**. Examples: AdaBoost, Gradient Boosting, XGBoost.
- **Stacking**: trains a meta-model to combine predictions from several different base models.

### Why Ensembles Work
- Aggregating diverse, individually-reasonable models reduces overfitting and variance.
- Effectiveness depends on base learners being better than random guessing and their errors being weakly correlated (the "diversity" principle).

This topic is primarily theoretical; the algorithms below (Random Forest, AdaBoost, Gradient Boosting, XGBoost) are concrete implementations of these ensemble strategies.

---

## Random Forest

### Introduction
**Random Forest** is a bagging-based ensemble of decision trees used for classification and regression. It builds many decorrelated trees and aggregates their outputs, improving accuracy and robustness over a single tree.

### Algorithm
1. Generate `N` bootstrap samples (sampling with replacement) from the training data.
2. Grow a decision tree on each bootstrap sample. At every split, only a **random subset of features** is considered (feature bagging), which decorrelates the trees.
3. Aggregate predictions:
   - **Classification**: majority vote across trees.
   - **Regression**: average of all tree outputs.

### Hyperparameters
| Hyperparameter | Description |
|---|---|
| `n_estimators` | Number of trees in the forest. |
| `max_depth` | Maximum depth of each tree. |
| `max_features` | Number of features considered at each split. |
| `min_samples_split` / `min_samples_leaf` | Minimum samples to split/form a leaf. |
| `bootstrap` | Whether bootstrap sampling is used when building trees. |
| `oob_score` | Whether to estimate accuracy using out-of-bag samples. |

### Example
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, max_features="sqrt", random_state=42)
rf.fit(X_train, y_train)

print("Accuracy:", rf.score(X_test, y_test))
```

---

## AdaBoost

### Introduction
**AdaBoost (Adaptive Boosting)** sequentially combines weak learners (commonly decision stumps — trees of depth 1) into a strong classifier. Each successive learner focuses more on samples that previous learners misclassified.

### Algorithm
1. Initialize equal weights for all training samples.
2. For each boosting round:
   - Train a weak learner on the weighted training data.
   - Compute the learner's weighted error rate.
   - Compute the learner's contribution weight (`alpha`): lower error → higher alpha.
   - Increase weights of misclassified samples and decrease weights of correctly classified ones.
3. Final prediction is a weighted vote (classification) or weighted sum (regression) of all weak learners.

### Hyperparameters
| Hyperparameter | Description |
|---|---|
| `n_estimators` | Number of weak learners (boosting rounds). |
| `learning_rate` | Shrinks the contribution of each weak learner. |
| `estimator` | Base weak learner (default: decision stump). |
| `algorithm` | Boosting variant used (e.g., SAMME) for multi-class classification. |

### Example
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42,
)
ada.fit(X_train, y_train)

print("Accuracy:", ada.score(X_test, y_test))
```

---

## Gradient Boosting

### Introduction
**Gradient Boosting** builds an ensemble of weak learners (typically shallow regression trees) sequentially, where each new learner is trained to predict the **residual errors** — formally, the negative gradient of the loss function — of the current ensemble.

### Algorithm
1. Initialize the model with a constant prediction (e.g., the mean of the target).
2. For each boosting round:
   - Compute pseudo-residuals: the negative gradient of the loss function with respect to current predictions.
   - Fit a weak learner (regression tree) to these pseudo-residuals.
   - Determine the optimal contribution (leaf values / step size).
   - Update the model: `F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)`
3. Final prediction is the sum of the initial model and all weighted weak learners.

### Hyperparameters
| Hyperparameter | Description |
|---|---|
| `n_estimators` | Number of boosting stages (trees). |
| `learning_rate` | Shrinks contribution of each tree; trades off with `n_estimators`. |
| `max_depth` | Maximum depth of each individual tree. |
| `subsample` | Fraction of samples used for fitting each tree (stochastic gradient boosting). |
| `min_samples_split` / `min_samples_leaf` | Minimum samples to split/form a leaf. |
| `loss` | Loss function to optimize (e.g., log-loss, squared error). |

### Example
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42
)
gb.fit(X_train, y_train)

print("Accuracy:", gb.score(X_test, y_test))
```

---

## XGBoost

### Introduction
**XGBoost (Extreme Gradient Boosting)** is an optimized, regularized implementation of gradient boosting built for speed and performance. It adds L1/L2 regularization, handles missing values natively, supports parallel and distributed computing, and uses second-order gradient information for more accurate tree construction.

### Algorithm
XGBoost extends gradient boosting with a regularized objective:

`Obj = Σ Loss(y_i, ŷ_i) + Σ Ω(f_k)`, where `Ω(f) = γT + (1/2)λ‖w‖²`

- `T` = number of leaves, `w` = leaf weights, `γ` and `λ` control model complexity.
- Uses a **second-order Taylor expansion** (gradients and Hessians) of the loss function to guide split selection, making tree construction more accurate than first-order gradient boosting.
- Employs shrinkage (learning rate), column (feature) subsampling, and efficient histogram-based / approximate split-finding algorithms for scalability.

### Hyperparameters
| Hyperparameter | Description |
|---|---|
| `n_estimators` | Number of boosting rounds (trees). |
| `learning_rate` (`eta`) | Step size shrinkage per boosting round. |
| `max_depth` | Maximum depth of each tree. |
| `subsample` | Fraction of training samples used per tree. |
| `colsample_bytree` | Fraction of features used per tree. |
| `gamma` | Minimum loss reduction required to make a further split. |
| `reg_alpha` / `reg_lambda` | L1 / L2 regularization on leaf weights. |
| `min_child_weight` | Minimum sum of instance weight needed in a child node. |

### Example
```python
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="mlogloss",
)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
```
