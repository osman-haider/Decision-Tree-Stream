# Decision-Tree-Stream

# Decision Stream: Deep, Merging Decision Trees for Classification and Regression

![Decision Stream vs Decision Tree](https://user-images.githubusercontent.com/your_image_path/decision_stream_fig.png)
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

**Decision Stream** is an advanced machine learning algorithm inspired by [this research paper](https://arxiv.org/abs/1704.07657) that improves on classical decision trees by introducing *statistical merging* of similar nodes.  
This creates a **deep, directed acyclic graph (DAG)** structure instead of a rigid tree, resulting in models that are more robust, less prone to overfitting, and perform better on complex, real-world datasets.

> **📰 Full Medium Article:**  
> [A Practical Guide to Decision Stream for Classification and Regression](https://medium.com/your-article-link)

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
git clone https://github.com/your-github-link.git
cd your-github-link
pip install -r requirements.txt
Dependencies:

numpy

scipy

scikit-learn

Usage
1. Classification Example (MNIST)

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
Or use our example script:

python src/main/classification.py
2. Regression Example (California Housing)

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
Or run:

python src/main/regression.py
Directory Structure

src/
  main/
    Decision_Stream.py       # Core Decision Stream class
    classification.py        # MNIST classification example
    regression.py            # California housing regression example
README.md
requirements.txt
Logging
This implementation uses Python's logging library. By default, all major events (splits, merges, terminal nodes) are logged to the console.
You can adjust the verbosity by editing logger.setLevel(logging.INFO) or logger.setLevel(logging.DEBUG) in Decision_Stream.py.

To log to a file instead, add these lines to your logger setup:

fh = logging.FileHandler('decision_stream.log')
fh.setFormatter(formatter)
logger.addHandler(fh)
References
Original Paper:
Decision Stream: Cultivating Deep Decision Trees (arXiv)

Project Article:
A Practical Guide to Decision Stream for Classification and Regression (Medium)

Related Libraries:

scikit-learn

numpy

scipy

Further Reading
Decision Trees Explained (scikit-learn docs)

Boosting and Bagging Decision Trees

License
This project is licensed under the MIT License.

If you find this repo useful, please ⭐️ it and share! Contributions and suggestions are welcome.

---

## **How to get a download link for this file**

1. **Manual way**:  
   - Copy the above contents into a file called `README.md` in your repo root.
   - [Optional] Upload your image to GitHub and replace the image URL at the top.
2. **Gist (quick download link):**  
   - Go to https://gist.github.com/
   - Paste the above into a new gist, name it `README.md`.
   - After saving, click "Raw" on the file for a direct download link.
3. **If you want me to generate a downloadable file here**:  
   - Let me know, and I'll create and upload it directly for you.

---
