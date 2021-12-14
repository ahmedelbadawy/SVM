import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



class SVM:
    def __init__(self, alpha=0.001, lambda_param=0.01, iters=1000):
        self.alpha = alpha
        self.lambda_param = lambda_param
        self.iters = iters
        self.w = None

    def fit(self, X, y):

        X = np.hstack((np.ones((len(X),1)),X))

        self.w = np.zeros(X.shape[1])
    
        for _ in range(self.iters):
            dist = 1-(y * (np.dot(X, self.w)))
            dist[dist<0] = 0
            for idx, x in enumerate(X):
                if dist[idx] == 0:
                    self.w -= self.alpha * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.alpha * ( 2 * self.lambda_param * self.w - np.dot(x, y[idx]))
    
    def compute_cost(self, X, y):
        X = np.hstack((np.ones((len(X),1)),X))
        distance = 1 - y * np.dot(X,self.w) 
        distance[distance<0] = 0
        cost = self.lambda_param * np.dot(self.w.transpose(),self.w) + (1/len(X)) * np.sum(distance)
        return cost


    def predict(self, X):
        X = np.hstack((np.ones((len(X),1)),X))
        return np.sign(np.dot(X, self.w))

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    Y = iris.target
    Y = np.where(Y == 0, -1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
    svm = SVM()
    svm.fit(X_train,Y_train)
    print(svm.compute_cost(X_test,Y_test))
    # print(accuracy_score(Y_test,svm.predict(X_test)))
    # plt.scatter(X[:,0],X[:,1],c = Y)
    # plt.show()