from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Decision_Stream import DecisionStream

# Load California Housing regression data
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Decision Stream
ds_reg = DecisionStream(plim=0.01, max_depth=12, min_samples_split=20, task='regression')
ds_reg.fit(X_train, y_train)
y_pred = ds_reg.predict(X_test)
print("California Housing Regression MSE:", mean_squared_error(y_test, y_pred))