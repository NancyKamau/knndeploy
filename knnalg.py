from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load and train the model inside the app
iris = load_iris()
X = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
