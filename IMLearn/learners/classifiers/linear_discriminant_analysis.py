from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_ = np.unique(y)
        cls_idx = {cls: j for j, cls in enumerate(self.classes_)}
        # pi
        self.pi_ = []
        for k in self.classes_:
            self.pi_.append(np.sum(y == k) / len(y))
        # mu
        mu_elems = [ [ list() for i in range(X.shape[1]) ] for j in range(self.classes_.size) ]
        design_sample_matrix = np.c_[X, y]
        for s in design_sample_matrix:
            j = cls_idx[s[-1]]
            for i, f in enumerate(s[:-1]):
                mu_elems[j][i].append(f)
        self.mu_ = np.ndarray((self.classes_.size, X.shape[1]))
        for j in range(self.classes_.size):
            for i in range(X.shape[1]):
                self.mu_[j][i] = sum(mu_elems[j][i]) / len(mu_elems[j][i])
        # cov
        sum_elems = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            sum_elems += np.outer((X[i] - self.mu_[cls_idx[y[i]]]),(X[i] - self.mu_[cls_idx[y[i]]]))
        self.cov_ = (1 / X.shape[0]) * sum_elems
        self._cov_inv = np.linalg.inv(self.cov_)


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

        y_cands = np.ndarray((self.classes_.size, X.shape[0]))
        for i in range(len(self.classes_)):
            a_i = self._cov_inv @ self.mu_[i]
            b_i = np.log(self.pi_[i]) - (float(self.mu_[i] @ self._cov_inv @ self.mu_[i])/2)
            y_cands[i] = a_i @ X.T + b_i
        k_max = np.argmax(y_cands, axis=0)
        return self.classes_[k_max]

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

        results = np.ndarray((X.shape[0], self.classes_.size))

        for i, x in enumerate(X):
            for j, k in enumerate(self.classes_):
                results[i][j] = (1 / np.sqrt(((2 * np.pi) ** X.shape[1]) * np.linalg.det(self.cov_))) * \
                     np.exp(-(1 / 2) * (x - self.mu_[j]).T @ self._cov_inv @ (x - self.mu_[j])) \
                * self.pi_[j]
        return results

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
