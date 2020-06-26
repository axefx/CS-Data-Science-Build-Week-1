from scipy.spatial import KDTree
from collections import Counter


class SimpleKnn:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.pred_neighbors = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.tree = KDTree(X, 30)
        # [5.1 3.5]
        return self

    def majority_vote(self, labels):
        """assumes that labels are ordered from nearest to farthest"""
        vote_counts = Counter(labels)
        winner, winner_count = vote_counts.most_common(1)[0]
        num_winners = len(
            [count for count in vote_counts.values() if count == winner_count])
        if num_winners == 1:
            return winner  # unique winner, so return it
        else:
            return majority_vote(labels[:-1])

    def predict(self, X):
        for point in X:
            self.pred_neighbors = (self.tree.query(
                point, k=self.n_neighbors, p=2))
        labels = [self.y[i] for i in self.pred_neighbors[1]]

        return [self.majority_vote(labels)]

    def kneighbors(self, X):
        if self.pred_neighbors:
            return self.pred_neighbors
        else:
            self.predict(X)
            return self.pred_neighbors


if __name__ == "__main__":
    from sklearn import datasets

    iris = datasets.load_iris()
    print(f"feature_names: {iris.feature_names[:2]}")
    print(f"target_names: {iris.target_names}")
    X = iris.data[:, :2]
    print(f'X: {len(X)}, {type(X)}')
    print(f'X head: {(X[:5])}, {type(X)}')
    y = iris.target
    print(f"y: {len(y)}, {type(y)}")

    print(type(KDTree))
    print(type(SimpleKnn))
    knn = SimpleKnn(n_neighbors=5)
    knn.fit(X, y)
    result = knn.predict([[5.1, 3.5]])
    print(result)
