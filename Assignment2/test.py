import sklearn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X, _ = sklearn.datasets.make_s_curve(n_samples=500)
X_data, y_data = X[:,0:2], X[:,2]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)
regressor = DecisionTreeRegressor().fit(X_train, y_train)
y_pred = regressor.predict(X_test)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_test[:,0], X_test[:,1], y_pred, label="Predictions")
ax.scatter(X_test[:,0], X_test[:,1], y_test, label="Labels")
plt.show()
