import numpy as np
from matplotlib import pyplot as plt
import random

random.seed(123)


class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Steps over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """

    def __init__(self, eta: float = 0.01, epochs: int = 10):
        self.eta = eta
        self.n_epochs = epochs
        self.w_ = None
        self.errors_ = None

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * self.backward(xi, target)
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def forward(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def backward(self, x, y):
        return y - self.predict(x)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.forward(X) >= 0.0, 1, -1)


class MLP:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = self.makeMatrix(self.ni, self.nh)
        self.wo = self.makeMatrix(self.nh, self.no)

        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = self.makeMatrix(self.ni, self.nh)
        self.co = self.makeMatrix(self.nh, self.no)

    @staticmethod
    def makeMatrix(I, J, fill=0.0):
        return np.zeros([I, J])

    @staticmethod
    def sigmoid(x):
        # return math.tanh(x)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(y):
        return y - y ** 2

    def backPropagate(self, targets, N, M):

        if len(targets) != self.no:
            print(targets)
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = np.zeros(self.no)
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = self.dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = np.zeros(self.nh)
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = self.dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change + M * self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        self.predict = np.empty([len(patterns), self.no])
        for i, p in enumerate(patterns):
            self.predict[i] = self.activate(p)
            # self.predict[i] = self.activate(p[0])

    def activate(self, inputs):

        if len(inputs) != self.ni - 1:
            print(inputs)
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum_h = 0.0
            for i in range(self.ni):
                sum_h += self.ai[i] * self.wi[i][j]
            self.ah[j] = self.sigmoid(sum_h)

        # output activations
        for k in range(self.no):
            sum_o = 0.0
            for j in range(self.nh):
                sum_o += self.ah[j] * self.wo[j][k]
            self.ao[k] = self.sigmoid(sum_o)

        return self.ao[:]

    def train(self, patterns, iterations=1000, N=0.5, M=0.1, printing=False):
        # N: learning rate
        # M: momentum factor
        patterns = list(patterns)
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.activate(inputs)
                error += self.backPropagate([targets], N, M)
            
            if (i % 5 == 0)&(printing==True):
                print('error in interation %d : %-.5f' % (i, error))
            if (printing==True):
                print('Final training error: %-.5f' % error)


# -------------------
# Utility Functions
# -------------------

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a


def load_data():
    """Load the test bed data for Perceptron"""
    data = np.genfromtxt('perceptron_toydata.txt', delimiter='\t')
    X, y = data[:, :2], data[:, 2]
    y = y.astype(np.int)
    return X, y


def plot_decision_boundary(ppn: Perceptron, X: tuple, y: tuple):
    """Plot Perceptron Decision Boundary"""

    w, b = ppn.weights, ppn.bias
    X_train, X_test = X
    y_train, y_test = y

    x_min = -2
    y_min = ((-(w[0] * x_min) - b[0])
             / w[1])

    x_max = 2
    y_max = ((-(w[0] * x_max) - b[0])
             / w[1])

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(10, 4))

    ax[0].plot([x_min, x_max], [y_min, y_max])
    ax[1].plot([x_min, x_max], [y_min, y_max])

    ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='class 0',
                  marker='o')
    ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='class 1',
                  marker='s')

    ax[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], label='class 0',
                  marker='o')
    ax[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], label='class 1',
                  marker='s')

    ax[1].legend(loc='upper right')
    return fig, ax


def plot_learning_step(i, weights, biases, X, y):
    """Plot Perceptron Learning Steps"""

    scatter_highlight_defaults = {'c': '',
                                  'edgecolor': 'k',
                                  'alpha': 1.0,
                                  'linewidths': 2,
                                  'marker': 'o',
                                  's': 150}

    fig, ax = plt.subplots()
    w, b = weights[i], biases[i]

    x_min = -20
    if w[1] != 0:
        y_min = ((-(w[0] * x_min) - b[0]) / w[1])
    else:
        y_min = 0

    x_max = 20
    if w[1] != 0:
        y_max = ((-(w[0] * x_max) - b[0]) / w[1])
    else:
        y_max = 0

    ax.plot([x_min, x_max], [y_min, y_max], color='k')

    ax.set_xlim([-5., 5])
    ax.set_ylim([-5., 5])

    ax.set_xlabel('Iteration %d' % (i + 1))

    ax.scatter(X[y == 0, 0], X[y == 0, 1], label='class 0',
               marker='o')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label='class 1',
               marker='s')

    ax.scatter(X[i][0], X[i][1], **scatter_highlight_defaults)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image
