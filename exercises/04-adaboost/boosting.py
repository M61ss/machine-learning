import matplotlib.pyplot as plt
import numpy as np

from utils import cmap

class WeakClassifier:
    """
    Function that models a WeakClassifier based on a simple threshold.
    """

    def __init__(self):

        # initialize a few stuff
        self._dim = None
        self._threshold = None
        self._label_above_split = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        n_samples, n_features = X.shape
        self._dim = np.random.choice(n_features)
        self._threshold = np.random.uniform(np.min(X[:, self._dim]), np.max(X[:, self._dim]))
        self._label_above_split = np.random.choice(np.unique(Y))

    def predict(self, X: np.ndarray):
        pred = np.ones(X.shape[0]) 
        pred[X[:, self._dim] < self._threshold] = self._label_above_split * -1
        pred[X[:, self._dim] >= self._threshold] = self._label_above_split
        return pred

class AdaBoostClassifier:
    """
    Function that models a Adaboost classifier
    """

    def __init__(self, n_learners: int, n_max_trials: int = 200):
        """
        Model constructor

        Parameters
        ----------
        n_learners: int
            number of weak classifiers.
        """

        # initialize a few stuff
        self.n_learners = n_learners
        self.learners = []
        self.alphas = np.zeros(shape=n_learners)
        self.n_max_trials = n_max_trials

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False):
        """
        Trains the model.

        Parameters
        ----------
        X: ndarray
            features having shape (n_samples, dim).
        Y: ndarray
            class labels having shape (n_samples,).
        verbose: bool
            whether or not to visualize the learning process.
            Default is False
        """

        n, d = X.shape
        possible_labels = np.unique(Y)

        if d != 2:
            verbose = False  # only plot learning if 2 dimensional

        assert possible_labels.size == 2, 'Error: data is not binary'

        for l in range(self.n_learners):
            # search for a weak classifier
            error = 1
            n_trials = 0
            cur_wclass = None
            y_pred = None

            if verbose:
                self._plot(cur_X, y_pred, sample_weights[cur_idx],
                           self.learners[-1], l)


    def predict(self, X: np.ndarray):
        """
        Function to perform predictions over a set of samples.

        Parameters
        ----------
        X: ndarray
            examples to predict. shape: (n_examples, d).

        Returns
        -------
        ndarray
            labels for each examples. shape: (n_examples,).

        """


    def _plot(self, X: np.ndarray, y_pred: np.ndarray, weights: np.ndarray,
              learner: WeakClassifier, iteration: int):

        # plot
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=weights * 50000,
                    cmap=cmap, edgecolors='k')

        M1, m1 = np.max(X[:, 1]), np.min(X[:, 1])
        M0, m0 = np.max(X[:, 0]), np.min(X[:, 0])

        cur_split = learner._threshold
        if learner._dim == 0:
            plt.plot([cur_split, cur_split], [m1, M1], 'k-', lw=5)
        else:
            plt.plot([m0, M0], [cur_split, cur_split], 'k-', lw=5)
        plt.xlim([m0, M0])
        plt.ylim([m1, M1])
        plt.xticks([])
        plt.yticks([])
        plt.title('Iteration: {:04d}'.format(iteration))
        plt.waitforbuttonpress(timeout=0.1)
