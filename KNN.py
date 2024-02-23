from scipy import stats
import numpy as np



class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.types = None
        self.X_test = None
        self.distance = None
        self.k_nearest = None
        self.y_pred = None
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.types = np.array(y_train)

    def predict(self, X_test):
        self.X_test = X_test
        self.distance = np.sum(np.square(self.X_train - self.X_test[:, np.newaxis]), axis=2)
        self.k_nearest = self.types[np.argpartition(self.distance, self.n_neighbors, axis=1)[:, :self.n_neighbors]]
        self.y_pred = stats.mode(self.k_nearest, axis=1).mode.reshape(self.X_test.shape[0])
        return self.y_pred