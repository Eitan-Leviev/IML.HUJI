from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        # classes
        self.classes_ = np.unique(y)
        # pi
        self.pi_ = []
        for k in self.classes_:
            self.pi_.append(np.sum(y == k) / len(y))
        # var & mu
        d = X.shape[1]
        clss = self.classes_.size
        self.mu_ = [ [ [] for i in range(d) ] for j in range(clss) ]
        class_indexes = {}
        self.vars_ = [ [ [] for i in range(d) ] for j in range(clss) ]

        # class dict
        for j, k in enumerate(self.classes_):
            class_indexes[k] = j
        design_sample_matrix = np.c_[X, y]
        for s in design_sample_matrix:

            i = class_indexes[s[-1]]
            for m, f in enumerate(s[:-1]):
                self.vars_[i][m].append(f)
                self.mu_[i][m].append(f)

        for i in range(clss):
            for j in range(d):
                self.vars_[i][j] = np.var(self.vars_[i][j])
                self.mu_[i][j] = sum(self.mu_[i][j]) / len(self.mu_[i][j])


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        y_pred = np.ndarray(X.shape[0])
        
        for j, s in enumerate(X):
            y = np.ndarray(self.classes_.size)
            for i, k in enumerate(self.classes_):
                y[i] = np.log(self.pi_[i]) - 0.5 * sum([np.log(2 * np.pi * self.vars_[i][j]) \
                              + ((s[j] - self.mu_[i][j]) ** 2) / self.vars_[i][j]
                                  for j in range(X.shape[1])])
            y_pred[j] = self.classes_[np.argmax(y)]

        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        def likelihood_expression(x, k):
            distribution = lambda x, i, k: (1 / np.sqrt(2 * np.pi * self.vars_[k][i])) * \
                np.exp(-0.5 * ((x[i] - self.mu_[k][i]) ** 2) / self.vars_[k][i])
            tmp = 1
            for i in range(X.shape[1]):
                tmp *= distribution(x, i, k)
            return tmp * self.pi_[k]
        likelihood_lst = np.ndarray((X.shape[0], self.classes_.size))
        for j, s in enumerate(X):
            for i, k in enumerate(self.classes_):
                likelihood_lst[j][i] = likelihood_expression(s, i)

        return likelihood_lst  # todo test

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error

        return misclassification_error(y, self._predict(X))
