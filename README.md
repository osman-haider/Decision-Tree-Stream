# Decision Stream: Deep, Merging Decision Trees for Classification and Regression

![Decision Stream vs Decision Tree](https://github.com/osman-haider/Decision-Tree-Stream/blob/master/images/dt_stream.png)
<!-- Replace the above with the raw link to the image you took from the paper and uploaded to your GitHub or an image host. -->

---

## Table of Contents

- [Overview](#overview)
- [How Decision Stream Works](#how-decision-stream-works)
- [Why Not Classic Decision Trees?](#why-not-classic-decision-trees)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Classification Example (MNIST)](#classification-example-mnist)
  - [Regression Example (California Housing)](#regression-example-california-housing)
- [Directory Structure](#directory-structure)
- [Logging](#logging)
- [References](#references)
- [Further Reading](#further-reading)
- [License](#license)

---

## Overview

**Decision Stream** is an advanced machine learning algorithm inspired by [this research paper](https://arxiv.org/pdf/1704.07657v3) that improves on classical decision trees by introducing *statistical merging* of similar nodes.  
This creates a **deep, directed acyclic graph (DAG)** structure instead of a rigid tree, resulting in models that are more robust, less prone to overfitting, and perform better on complex, real-world datasets.

> **📰 Full Medium Article:**  
> [A Practical Guide to Decision Stream for Classification and Regression](https://usman-haider.medium.com/beyond-decision-trees-a-practical-guide-to-decision-stream-for-classification-and-regression-with-4b2c8c6c4102)

---

## How Decision Stream Works

1. **Split:** Like a traditional decision tree, nodes are split based on the best feature/value.
2. **Merge:** After every split, leaf nodes are tested for statistical similarity (using t-test, Kolmogorov-Smirnov, or Mann-Whitney U test). If two leaves are statistically similar, they are **merged**.
3. **Repeat:** This split-then-merge cycle continues until stopping criteria are reached (e.g., all leaves are terminal or maximum depth).
4. **Predict:** For inference, the graph is traversed from root to terminal nodes, similar to a tree but now possibly passing through merged branches.

**Benefits:**
- Reduces overfitting by keeping more samples in each node.
- Produces deeper, narrower, and more generalizable models.
- Retains interpretability while allowing for flexible, adaptive structure.

---

## Why Not Classic Decision Trees?

- **Overfitting:** Classic trees quickly create small, unreliable leaves by aggressive splitting.
- **No Fusion:** Once data is split, the tree cannot merge back similar groups.
- **Poor Generalization:** Especially for deep or high-dimensional problems, classic trees lose predictive power on new data.

**Decision Stream** addresses these by dynamically merging statistically similar branches, producing a model that's both deep and robust.

---

## Key Features

- Works for **both classification and regression** tasks.
- Fully written in **Python** with **scikit-learn compatible** interface.
- **Built-in logging** for transparency and experiment tracking.
- Easy to use and adapt to your own datasets.

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/osman-haider/Decision-Tree-Stream
cd Decision-Tree-Stream
pip install -r requirements.txt
```

Dependencies:
- numpy
- scipy
- scikit-learn

---

## Usage

### **1. Classification Example (MNIST)**

```python
from src.main.Decision_Stream import DecisionStream
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True)
X = X / 255.0
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X[:5000], y[:5000], test_size=0.2, random_state=42)

ds_clf = DecisionStream(plim=0.005, max_depth=10, min_samples_split=30, task='classification')
ds_clf.fit(X_train, y_train)
y_pred = ds_clf.predict(X_test)
print("MNIST (Subset) Classification Accuracy:", accuracy_score(y_test, y_pred))
```

Or use our example script:

```bash
python src/main/classification.py
```

---

### **2. Regression Example (California Housing)**

```python
from src.main.Decision_Stream import DecisionStream
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ds_reg = DecisionStream(plim=0.01, max_depth=12, min_samples_split=20, task='regression')
ds_reg.fit(X_train, y_train)
y_pred = ds_reg.predict(X_test)
print("California Housing Regression MSE:", mean_squared_error(y_test, y_pred))
```

Or run:

```bash
python src/main/regression.py
```

---

## Directory Structure

```
src/
  main/
    Decision_Stream.py       # Core Decision Stream class
    classification.py        # MNIST classification example
    regression.py            # California housing regression example
README.md
requirements.txt
```

---

## Logging

This implementation uses Python's `logging` library. By default, all major events (splits, merges, terminal nodes) are logged to the console.  
You can adjust the verbosity by editing `logger.setLevel(logging.INFO)` or `logger.setLevel(logging.DEBUG)` in `Decision_Stream.py`.

To log to a file instead, add these lines to your logger setup:

```python
fh = logging.FileHandler('decision_stream.log')
fh.setFormatter(formatter)
logger.addHandler(fh)
```

---

## References

- **Original Paper:**  
  [Decision Stream: Cultivating Deep Decision Trees (arXiv)](https://arxiv.org/pdf/1704.07657v3)
- **Project Article:**  
  [A Practical Guide to Decision Stream for Classification and Regression (Medium)](https://usman-haider.medium.com/beyond-decision-trees-a-practical-guide-to-decision-stream-for-classification-and-regression-with-4b2c8c6c4102)
- **Related Libraries:**  
  - [scikit-learn](https://scikit-learn.org/)
  - [numpy](https://numpy.org/)
  - [scipy](https://scipy.org/)

---

## Further Reading

- [Decision Trees Explained (scikit-learn docs)](https://scikit-learn.org/stable/modules/tree.html)
- [Boosting and Bagging Decision Trees](https://scikit-learn.org/stable/modules/ensemble.html)

---

## License

This project is licensed under the MIT License.

---

**If you find this repo useful, please ⭐️ it and share! Contributions and suggestions are welcome.**
