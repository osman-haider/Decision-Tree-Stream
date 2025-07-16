from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Decision_Stream import DecisionStream

# Load MNIST (will auto-download)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist['data'] / 255.0  # Normalize pixels
y = mnist['target'].astype(int)

# For quick testing, use a subset (change as needed)
X_train, X_test, y_train, y_test = train_test_split(X[:5000], y[:5000], test_size=0.2, random_state=42)

# Train and evaluate Decision Stream
ds_clf = DecisionStream(plim=0.005, max_depth=12, min_samples_split=30, task='classification')
ds_clf.fit(X_train, y_train)
y_pred = ds_clf.predict(X_test)
print("MNIST (Subset) Classification Accuracy:", accuracy_score(y_test, y_pred))
