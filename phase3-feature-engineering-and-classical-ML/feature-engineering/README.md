# Feature Engineering for Tabular Data — Reference

A structured reference covering the core feature engineering techniques demonstrated in the numbered notebooks of this directory. Each section gives the core idea, the governing formula(s), and a brief code snippet showing the canonical usage. The numbered notebooks themselves are hands-on demos and should be consulted for full context.

---

## Table of Contents

1. [Feature Scaling](#1-feature-scaling)
2. [Handling Categorical Data](#2-handling-categorical-data)
3. [Encoding Numerical Features](#3-encoding-numerical-features)
4. [Feature Transformations](#4-feature-transformations)
5. [Handling Mixed Variables](#5-handling-mixed-variables)
6. [Handling Date and Time Features](#6-handling-date-and-time-features)
7. [Dimensionality Reduction — PCA](#7-dimensionality-reduction--pca)
8. [Handling Outliers](#8-handling-outliers)
9. [Handling Missing Values](#9-handling-missing-values)
10. [Column Transformer and ML Pipelines](#10-column-transformer-and-ml-pipelines)

---

## 1. Feature Scaling

Bring features to a comparable magnitude so that distance- and gradient-based models are not dominated by high-range columns.

### 1.1 Standardization (Z-score)

Centers each feature to mean 0 and rescales it to unit variance.

$$
x' = \frac{x - \mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn μ, σ on train
X_test_scaled  = scaler.transform(X_test)       # apply same parameters
```

### 1.2 Min–Max Scaling (Normalization)

Rescales each feature to a fixed range, default `[0, 1]`.

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**When to use what:** Standardization works for most cases and is robust to small outliers. Min–Max is preferred when you need bounded values (e.g. neural networks with sigmoid/tanh, image pixel data), but is sensitive to outliers.

---

## 2. Handling Categorical Data

Categorical data falls into two families: **nominal** (no order, e.g. color) and **ordinal** (natural order, e.g. Poor < Average < Good).

### 2.1 Ordinal Encoding

Map each category to a single integer that preserves the order.

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['Poor', 'Average', 'Good'],
                                ['School', 'UG', 'PG']])
X_encoded = oe.fit_transform(X)   # Poor→0, Average→1, Good→2, ...
```

### 2.2 Label Encoding (target only)

Encode the target label `y` to integers 0..K−1.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

### 2.3 One-Hot Encoding (OHE)

Create one binary column per category. For `k` categories, this expands a column into `k` binary columns (or `k − 1` if `drop_first=True` to avoid the dummy-variable trap / multicollinearity).

```python
import pandas as pd
df = pd.get_dummies(df, columns=['fuel', 'owner'], drop_first=True)

# or via scikit-learn
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_enc = ohe.fit_transform(X[['fuel', 'owner']])
```

### 2.4 OHE with Top Categories (high cardinality)

When a category has many levels (e.g. car brands), collapse rare levels into a single `"uncommon"` bucket to keep the feature space manageable.

```python
counts = df['brand'].value_counts()
threshold = 100
repl = counts[counts <= threshold].index
df['brand'] = df['brand'].replace(repl, 'uncommon')
df = pd.get_dummies(df, columns=['brand'], drop_first=True)
```

---

## 3. Encoding Numerical Features

### 3.1 Discretization / Binning

Convert a continuous variable into a discrete one by mapping values to bins. Useful for handling outliers, smoothing value spread, and letting tree-based models learn step-functions.

**Strategies:**

- **Uniform (equal width):** bins of equal range.
- **Quantile (equal frequency):** each bin has (approximately) the same number of points.
- **K-means:** bin edges are cluster centroids.
- **Decision tree (supervised):** bins chosen to maximize information gain with respect to `y`.

```python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer

trf = ColumnTransformer([
    ('age',  KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile'), [0]),
    ('fare', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile'), [1]),
])
X_binned = trf.fit_transform(X)
```

### 3.2 Binarization

Threshold a numeric feature into 0/1 using a single cutoff.

$$
x' = \begin{cases} 1 & \text{if } x > \text{threshold} \\ 0 & \text{otherwise} \end{cases}
$$

```python
from sklearn.preprocessing import Binarizer
from sklearn.compose import ColumnTransformer

trf = ColumnTransformer([
    ('bin', Binarizer(threshold=0.5, copy=False), ['family'])
], remainder='passthrough')
X_bin = trf.fit_transform(X)
```

---

## 4. Feature Transformations

Many models (linear/logistic regression, KNN, PCA, neural nets) work better when features are roughly Gaussian. Transformations reshape skewed distributions to be more symmetric.

### 4.1 Function Transforms

| Transform | Formula | Notes |
|-----------|---------|-------|
| Log | $\log(1 + x)$ | Right-skewed data, no negatives |
| Reciprocal | $1/x$ | Strong tail compression |
| Square | $x^2$ | Left-skewed data |
| Square-root | $\sqrt{x}$ | Mild right skew, no negatives |

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

trf = ColumnTransformer([
    ('log', FunctionTransformer(np.log1p), ['Fare'])
], remainder='passthrough')
X_t = trf.fit_transform(X)
```

### 4.2 Power Transforms

Learn the best exponent $\lambda$ that makes the data most Gaussian. Two flavors are available:

**Box-Cox** (requires strictly positive $y$):

$$
y^{(\lambda)} =
\begin{cases}
\dfrac{y^\lambda - 1}{\lambda}, & \lambda \neq 0 \\[6pt]
\log(y), & \lambda = 0
\end{cases}
$$

**Yeo-Johnson** (handles zero and negative values):

$$
y^{(\lambda)} =
\begin{cases}
\dfrac{(y+1)^\lambda - 1}{\lambda}, & y \geq 0,\ \lambda \neq 0 \\
\log(y+1), & y \geq 0,\ \lambda = 0 \\[6pt]
\dfrac{-\big[(-y+1)^{2-\lambda} - 1\big]}{2-\lambda}, & y < 0,\ \lambda \neq 2 \\
-\log(-y+1), & y < 0,\ \lambda = 2
\end{cases}
$$

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox (data must be strictly positive)
pt = PowerTransformer(method='box-cox', standardize=True)
X_t = pt.fit_transform(X + 1e-6)

# Yeo-Johnson (supports zeros and negatives)
pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_t = pt.fit_transform(X)
```

**How to diagnose non-normality:** `sns.histplot` / `sns.kdeplot`, `df.skew()`, or — most reliably — a **QQ plot** (`scipy.stats.probplot(x, dist="norm", plot=plt)`).

---

## 5. Handling Mixed Variables

A single column can contain both a numeric and a categorical component (e.g. `Cabin = "C85"`, `Ticket = "PC 17599"`, `number = "5A"`). Split the column into two — one numeric, one categorical — and process each appropriately.

```python
import numpy as np

# extract numeric and categorical parts of a mixed column
df['number_numerical']   = pd.to_numeric(df['number'], errors='coerce')
df['number_categorical'] = np.where(df['number_numerical'].isnull(),
                                    df['number'], np.nan)

# regex extraction (e.g., digits inside "C85")
df['cabin_num'] = df['Cabin'].str.extract(r'(\d+)')
df['cabin_cat'] = df['Cabin'].str[0]   # first character
```

---

## 6. Handling Date and Time Features

Datetime columns are usually dropped raw. Extract structured features that capture cyclical/seasonal patterns, then let the model learn from them.

```python
import numpy as np
import pandas as pd

df['date'] = pd.to_datetime(df['date'])   # parse to datetime64

df['year']         = df['date'].dt.year
df['month']        = df['date'].dt.month
df['month_name']   = df['date'].dt.month_name()
df['day']          = df['date'].dt.day
df['dayofweek']    = df['date'].dt.dayofweek
df['day_name']     = df['date'].dt.day_name()
df['is_weekend']   = df['day_name'].isin(['Saturday', 'Sunday']).astype(int)
df['quarter']      = df['date'].dt.quarter
df['semester']     = np.where(df['quarter'].isin([1, 2]), 1, 2)

# elapsed time (e.g. age of a record)
import datetime
today = datetime.datetime.today()
df['days_since']   = (today - df['date']).dt.days

# from a time-of-day column
df['hour'] = df['date'].dt.hour
df['min']  = df['date'].dt.minute
df['time'] = df['date'].dt.time
```

---

## 7. Dimensionality Reduction — PCA

Principal Component Analysis projects the data onto orthogonal axes (principal components) that capture the maximum variance. Use it to compress many correlated features into a few uncorrelated ones for visualization, noise reduction, or to speed up models.

**Algorithm:**

1. Standardize features (mean 0, unit variance).
2. Compute the covariance matrix $\Sigma$.
3. Compute eigenvectors (principal axes) and eigenvalues (variance explained) of $\Sigma$.
4. Sort components by descending eigenvalue; keep the top $k$.
5. Project: $X_{\text{new}} = X \cdot W_k$, where $W_k$ holds the top $k$ eigenvectors.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)         # 1) scale
pca   = PCA(n_components=2)
X_pc  = pca.fit_transform(X_std)                 # 2-5) fit + project
print(pca.explained_variance_ratio_)             # variance per component
```

---

## 8. Handling Outliers

Outliers distort means, variances, and distance-based models. Three common detection rules:

### 8.1 Z-Score Method

Assumes the feature is roughly normal. A point is an outlier if its absolute z-score exceeds a threshold (typically 3).

$$
z = \frac{x - \mu}{\sigma}, \qquad \text{outlier if } |z| > 3
$$

```python
# flag / trim
df['z'] = (df['x'] - df['x'].mean()) / df['x'].std()
df_clean = df[df['z'].abs() < 3]                # trimming

# cap (winsorize)
upper = df['x'].mean() + 3 * df['x'].std()
lower = df['x'].mean() - 3 * df['x'].std()
df['x'] = df['x'].clip(lower, upper)            # capping
```

### 8.2 IQR Method (Tukey's fences)

Distribution-free. Outliers are points that fall outside the "fences":

$$
\text{lower} = Q_1 - 1.5 \cdot IQR, \qquad
\text{upper} = Q_3 + 1.5 \cdot IQR, \qquad
IQR = Q_3 - Q_1
$$

```python
q1, q3 = df['x'].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

df_clean = df[(df['x'] >= lower) & (df['x'] <= upper)]   # trim
df['x']  = df['x'].clip(lower, upper)                     # cap
```

### 8.3 Percentile Method

Use when you want to ignore a fixed fraction at each tail (e.g. bottom 1% and top 1%). Robust to extreme outliers because the limits come from the data itself.

```python
lower = df['x'].quantile(0.01)
upper = df['x'].quantile(0.99)

df_clean = df[(df['x'] >= lower) & (df['x'] <= upper)]   # trim
df['x']  = df['x'].clip(lower, upper)                     # cap (Winsorization)
```

**Trim vs. cap:** Trimming drops rows (loses information but keeps the rest of the data clean). Capping (winsorization) keeps the row count but replaces extreme values with the boundary — useful when outliers are genuine measurement errors or rare but real cases.

---

## 9. Handling Missing Values

Two high-level strategies: **remove** rows/columns, or **impute** (fill) the missing entries.

### 9.1 Complete Case Analysis (CCA) — drop rows

Drop any row that has at least one missing value. Safe only when the data is **MCAR** (Missing Completely At Random) and the missing rate per column is low (rule of thumb: < 5%).

```python
cols = [c for c in df.columns
        if 0 < df[c].isnull().mean() < 0.05]   # low-missing columns
new_df = df[cols].dropna()
```

### 9.2 Univariate Imputation (single column at a time)

**Numerical data:**

| Strategy | When to use |
|----------|-------------|
| `mean` | Roughly symmetric, no outliers |
| `median` | Skewed or has outliers |
| `constant` (arbitrary, e.g. `-1`, `999`) | You want missingness to be a separate signal; works well with trees |
| Random sample | Preserves the original distribution |

**Categorical data:**

| Strategy | When to use |
|----------|-------------|
| `most_frequent` (mode) | General default |
| `constant` (`"Missing"`) | When "missing" is itself informative |
| Random sample | Preserves the original distribution |

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

trf = ColumnTransformer([
    ('num',  SimpleImputer(strategy='median'),               ['Age', 'Fare']),
    ('cat',  SimpleImputer(strategy='most_frequent'),        ['Embarked']),
    ('arb',  SimpleImputer(strategy='constant', fill_value=-1), ['Cabin']),
], remainder='passthrough')

X_imp = trf.fit_transform(X)
```

**Random sample imputation** (preserves distribution, but you implement it yourself):

```python
X['Age_imputed'] = X['Age']
mask = X['Age'].isnull()
X.loc[mask, 'Age_imputed'] = X['Age'].dropna().sample(mask.sum()).values
```

### 9.3 Missing Indicator

Add a binary column `X_was_missing` that flags whether the value was originally missing — this lets the model learn from the *fact* of being missing, not just from the imputed value.

```python
from sklearn.impute import SimpleImputer, MissingIndicator

# 1) explicit indicator
mi = MissingIndicator()
X_ind = mi.fit_transform(X)             # boolean mask of missing cells
X_aug = np.hstack([X, X_ind.astype(int)])

# 2) convenience: ask the imputer to add it for you
si = SimpleImputer(strategy='median', add_indicator=True)
X_aug = si.fit_transform(X)
```

### 9.4 Automatic Parameter Selection for Imputation

Use `GridSearchCV` over a `Pipeline` to pick the best imputation strategy (mean vs median, most-frequent vs constant, etc.) along with the rest of the model.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer()), ('sc', StandardScaler())]),
             ['Age', 'Fare']),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore'))]),
             ['Embarked', 'Sex']),
])

clf = Pipeline([('pre', pre), ('lr', LogisticRegression())])

grid = GridSearchCV(clf, {
    'pre__num__imp__strategy': ['mean', 'median'],
    'pre__cat__imp__strategy': ['most_frequent', 'constant'],
    'lr__C': [0.1, 1.0, 10],
}, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

### 9.5 Multivariate Imputation

Use the *other* features to predict the missing one — usually outperforms univariate methods.

**KNN Imputer** — fill `x_i` with the weighted average of `x_i` over the `k` nearest neighbors (in feature space) that do have a value for `x_i`.

```python
from sklearn.impute import KNNImputer

imp = KNNImputer(n_neighbors=5, weights='distance')
X_imp = imp.fit_transform(X)
```

**MICE / Iterative Imputer** — for each feature with missing values, treat it as the target `y`, fit a regression model on the remaining features, predict the missing entries, then iterate over features until convergence.

```python
from sklearn.experimental import enable_iterative_imputer      # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
X_imp = imp.fit_transform(X)
```

---

## 10. Column Transformer and ML Pipelines

Real datasets mix numeric, categorical, and missing-valued columns, each needing a different transformer. The two scikit-learn primitives below keep that logic clean, reproducible, and free of leakage.

### 10.1 `ColumnTransformer`

Apply different transformers to different columns in a single call, and concatenate the results. The `remainder='passthrough'` option keeps un-touched columns; `remainder='drop'` (default) discards them.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

trf = ColumnTransformer(transformers=[
    ('impute_age',  SimpleImputer(),                          ['Age']),
    ('impute_emb',  SimpleImputer(strategy='most_frequent'),  ['Embarked']),
    ('ohe',         OneHotEncoder(drop='first', handle_unknown='ignore'),
                                                            ['Sex', 'Embarked']),
    ('scale',       StandardScaler(),                         ['Age', 'Fare']),
], remainder='passthrough')

X_t = trf.fit_transform(X_train)
```

### 10.2 `Pipeline` (and `make_pipeline`)

Chain transformers and a final estimator so the same preprocessing is applied to every fold of cross-validation and to any new data at predict time — preventing data leakage.

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier

# explicit names — useful for GridSearchCV
pipe = Pipeline([
    ('trf1', ColumnTransformer([...])),
    ('trf2', ColumnTransformer([...])),
    ('scale', StandardScaler()),
    ('select', SelectKBest(score_func=chi2, k=5)),
    ('clf',  DecisionTreeClassifier()),
])

# shorter form when names aren't needed
pipe = make_pipeline(trf1, trf2, StandardScaler(), SelectKBest(k=5), DecisionTreeClassifier())

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

Because preprocessing and the model live in one object, cross-validation, grid search, and serialization all "just work":

```python
from sklearn.model_selection import cross_val_score
print(cross_val_score(pipe, X_train, y_train, cv=5).mean())

import pickle
with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)
# later:
pipe = pickle.load(open('pipe.pkl', 'rb'))
pipe.predict(new_data)        # preprocessing is automatic
```

---

## Suggested Workflow

Putting it all together, a typical feature-engineering pipeline on a fresh tabular dataset proceeds in this order:

1. **Parse & extract** — convert types, split mixed columns, extract date/time components.
2. **Handle missing values** — decide drop vs. simple vs. multivariate imputation; add a missing indicator when "missing" is informative.
3. **Encode categoricals** — ordinal encoding for ordered levels, one-hot for nominal; collapse rare levels.
4. **Encode numerics** — discretize/binarize when useful; otherwise leave continuous.
5. **Transform distributions** — apply log / power transforms to reduce skew if using a linear model.
6. **Handle outliers** — trim or cap with z-score / IQR / percentile rules.
7. **Scale** — standardize for distance-/gradient-based models.
8. **Reduce** — PCA when many correlated features or for visualization.
9. **Wrap in a pipeline** — combine every step with the model in a `Pipeline` so cross-validation and prediction are leak-free.

Each numbered notebook in this directory demonstrates one of these steps in detail — open the matching notebook to see the full demo.
