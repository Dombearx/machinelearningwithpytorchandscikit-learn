import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm


class Net:

    def __init__(self, lr=0.01, number_of_epoch=100, start_state=1):
        self.lr = lr
        self.number_of_epoch = number_of_epoch
        self.start_state = start_state

    def fit(self, X, y):
        np.random.seed(self.start_state)

        self.bias = 0
        self.weights = np.random.rand(X.shape[1])
        self.errors = []


        for epoch in range(self.number_of_epoch):
            epoch_errors = 0
            for x, target in zip(X, y):
                bias_update = self.lr * (target - self.predict(x))
                weight_update = bias_update * x
                if bias_update != 0:
                    epoch_errors += 1

                self.bias += bias_update
                self.weights += weight_update
            self.errors.append(epoch_errors)
            print(f"{epoch = } {epoch_errors =}")
        print(self.weights)

    def predict(self, X):
        res = np.sum((self.weights * X)) + self.bias
        return 1 if res > 0 else 0

    def predict_mass(self, X):
        res = []
        for x in X:
            res.append(self.predict(x))

        return np.array(res)

    def show_loss(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of errors')

        # plt.savefig('images/02_07.png', dpi=300)
        plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict_mass(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    plt.show()

def read_data():
    df = pd.read_csv(
        'iris.data',
        header=None, encoding='utf-8')
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    X = df.iloc[0:100, [0, 2]].values
    return X, y
if __name__ == '__main__':
    # X = np.array([[1, 2, 3], [1, 2, 2], [2, 3, 3]])
    # y = np.array([1, 0, 1])
    # n = Net(number_of_epoch=100)
    # n.fit(X, y)
    # print(f"{n.predict([1, 2, 3]) = }")
    # print(f"{n.predict([1, 2, 2]) = }")
    # print(f"{n.predict([2, 3, 3]) = }")

    X, y = read_data()

    plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 's', label = 'Versicolor')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

    n = Net(lr=0.1, number_of_epoch=10)
    n.fit(X, y)
    n.show_loss()

    plot_decision_regions(X, y, n)





