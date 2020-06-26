from sklearn import datasets, neighbors
from knn import SimpleKnn

iris = datasets.load_iris()
print(f"feature_names: {iris.feature_names[:2]}")
print(f"target_names: {iris.target_names}")
X = iris.data[:, :2]
print(f'X: {len(X)}, {type(X)}')
print(f'X head: {(X[:5])}, {type(X)}')
y = iris.target
print(f"y: {len(y)}, {type(y)}")
d = dict()
for i in y:
    if i in d:
        d[i] += 1
    else:
        d[i] = 1
print(f"y counts: {d}, {type(d)}")

n_neighbors = 15
clf = neighbors.KNeighborsClassifier(
    n_neighbors=n_neighbors, algorithm='kd_tree')
clf.fit(X, y)
print(clf.get_params())
results = clf.predict([[4.8, 3.3]])
print(results)
nearest = clf.kneighbors([[4.8, 3.3]])
nearest_neighbors = []
for each in nearest[1][0]:
    nearest_neighbors.append((X[each], y[each]))
print(nearest_neighbors)
# [0]
# [(array([4.8, 3.4]), 0), (array([4.8, 3.4]), 0), (array([4.7, 3.2]), 0),
# (array([4.7, 3.2]), 0), (array([4.8, 3.1]), 0), (array([5. , 3.3]), 0),
# (array([4.9, 3.1]), 0), (array([4.6, 3.2]), 0), (array([4.9, 3.1]), 0),
# (array([5. , 3.2]), 0), (array([4.6, 3.4]), 0), (array([5. , 3.4]), 0),
# (array([5. , 3.4]), 0), (array([4.6, 3.1]), 0), (array([5. , 3.5]), 0)]

n_neighbors = 15
clf = SimpleKnn(n_neighbors=n_neighbors)
clf.fit(X, y)
results = clf.predict([[4.8, 3.3]])
print(results)
nearest = clf.kneighbors([[4.8, 3.3]])
nearest_neighbors = []
for each in nearest[1]:
    nearest_neighbors.append((X[each], y[each]))
print(nearest_neighbors)

# [(array([5.1, 3.5]), 0), (array([5.1, 3.5]), 0), (array([5. , 3.5]), 0), (array([5. , 3.5]), 0), (array([5.1, 3.4]), 0)]
print("--------"*10)
n_neighbors = 5
clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X, y)
nearest = clf.kneighbors([[5.1, 3.5]])
nearest_neighbors = []
for each in nearest[1][0]:
    nearest_neighbors.append((X[each], y[each]))
print(nearest_neighbors)
