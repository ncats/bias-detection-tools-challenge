# -*- coding: utf-8 -*-
# Based on sklearn.ensemble.AdaBoostClassifier
from inspect import isclass
import numbers
import warnings
import math

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
from scipy.special import xlogy

warnings.simplefilter(action='ignore', category=FutureWarning)


def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
    log_proba = np.log(proba)

    return (n_classes - 1) * (
        log_proba - (1.0 / n_classes) * log_proba.sum(axis=1)[:, np.newaxis]
    )


def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to specify if the
    estimator is fitted or not.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Raises
    ------
    TypeError
        If the estimator is a class or not an estimator instance

    NotFittedError
        If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        fitted = all_or_any([hasattr(estimator, attr) for attr in attributes])
    elif hasattr(estimator, "__sklearn_is_fitted__"):
        fitted = estimator.__sklearn_is_fitted__()
    else:
        fitted = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    if not fitted:
        raise Exception(msg % {"name": type(estimator).__name__})


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
            The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


class FABulous(ClassifierMixin):
    """An AdaBoost classifier.

    An AdaBoost [1] classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm known as AdaBoost-SAMME [2].

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper
            ``classes_`` and ``n_classes_`` attributes. If ``None``, then
            the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
            initialized with `max_depth=1`.

            .. versionadded:: 1.2
               `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.
            Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
            Weight applied to each classifier at each boosting iteration. A higher
            learning rate increases the contribution of each classifier. There is
            a trade-off between the `learning_rate` and `n_estimators` parameters.
            Values must be in the range `(0.0, inf)`.

    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME.R'
            If 'SAMME.R' then use the SAMME.R real boosting algorithm.
            ``estimator`` must support calculation of class probabilities.
            If 'SAMME' then use the SAMME discrete boosting algorithm.
            The SAMME.R algorithm typically converges faster than SAMME,
            achieving a lower test error with fewer boosting iterations.

    random_state : int, RandomState instance or None, default=None
            Controls the random seed given at each `estimator` at each
            boosting iteration.
            Thus, it is only used when `estimator` exposes a `random_state`.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

    base_estimator : object, default=None
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper
            ``classes_`` and ``n_classes_`` attributes. If ``None``, then
            the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
            initialized with `max_depth=1`.

            .. deprecated:: 1.2
                    `base_estimator` is deprecated and will be removed in 1.4.
                    Use `estimator` instead.

    Attributes
    ----------
    estimator_ : estimator
            The base estimator from which the ensemble is grown.

            .. versionadded:: 1.2
               `base_estimator_` was renamed to `estimator_`.

    base_estimator_ : estimator
            The base estimator from which the ensemble is grown.

            .. deprecated:: 1.2
                    `base_estimator_` is deprecated and will be removed in 1.4.
                    Use `estimator_` instead.

    estimators_ : list of classifiers
            The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,)
            The classes labels.

    n_classes_ : int
            The number of classes.

    estimator_weights_ : ndarray of floats
            Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
            Classification error for each estimator in the boosted
            ensemble.

    feature_importances_ : ndarray of shape (n_features,)
            The impurity-based feature importances if supported by the
            ``estimator`` (when based on decision trees).

            Warning: impurity-based feature importances can be misleading for
            high cardinality features (many unique values). See
            :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
            Number of features seen during :term:`fit`.

            .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during :term:`fit`. Defined only when `X`
            has feature names that are all strings.

            .. versionadded:: 1.0

    See Also
    --------
    AdaBoostRegressor : An AdaBoost regressor that begins by fitting a
            regressor on the original dataset and then fits additional copies of
            the regressor on the same dataset but where the weights of instances
            are adjusted according to the error of the current prediction.

    GradientBoostingClassifier : GB builds an additive model in a forward
            stage-wise fashion. Regression trees are fit on the negative gradient
            of the binomial or multinomial deviance loss function. Binary
            classification is a special case where only a single regression tree is
            induced.

    sklearn.tree.DecisionTreeClassifier : A non-parametric supervised learning
            method used for classification.
            Creates a model that predicts the value of a target variable by
            learning simple decision rules inferred from the data features.

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
               on-Line Learning and an Application to Boosting", 1995.

    .. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    AdaBoostClassifier(n_estimators=100, random_state=0)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    >>> clf.score(X, y)
    0.983...
    """

    def __init__(
            self,
            prot_col, ref_col, fairness_criteria,
            n_estimators=50,
            learning_rate=1.0,
            algorithm="SAMME.R",
            random_state=None
    ):
        self.prot_col = prot_col
        self.ref_col = ref_col
        self.fairness_criteria = fairness_criteria
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state

    def fairyfy(self, prot_col, ref_col, X_train, y_train, y_pred,
                fairness_criteria, confusion_matrices):
        cfs = confusion_matrices
        b = np.ones(len(X_train))
        if fairness_criteria == 'get_equalized_odds':
            b = math.sqrt((cfs[1][1, 1]*cfs[0][1, 0]) /
                          (cfs[0][1, 1]*cfs[1][1, 0]))
            b_ = math.sqrt((cfs[1][0, 0]*cfs[0][0, 1]) /
                           (cfs[0][0, 0]*cfs[1][0, 1]))
            b[(X_train[prot_col] == 1) & (
                y_train == 0) & (y_pred == 1)] = b
            b[(X_train[prot_col] == 0) & (
                y_train == 0) & (y_pred == 1)] = 1/b
            b[(X_train[prot_col] == 1) & (
                y_train == 1) & (y_pred == 0)] = b_
            b[(X_train[prot_col] == 0) & (
                y_train == 1) & (y_pred == 0)] = 1/b_
        elif fairness_criteria == 'get_equal_opportunity':
            b = math.sqrt((cfs[1][0, 0]*cfs[0][0, 1]) /
                          (cfs[0][0, 0]*cfs[1][0, 1]))
            b[(X_train[prot_col] == 1) & (
                y_train == 1) & (y_pred == 0)] = b
            b[(X_train[prot_col] == 0) & (
                y_train == 1) & (y_pred == 0)] = 1/b
        elif fairness_criteria == 'get_demographic_parity':
            b = math.sqrt((cfs[0][0, 0]*cfs[0][1, 0])/(cfs[0][1, 1]*cfs[0][0, 1]) +
                          (cfs[1][0, 0]*cfs[1][1, 0])/(cfs[1][1, 1]*cfs[1][0, 1]))
            b[(X_train[prot_col] == 1) & (y_pred == 1)] = b
            b[(X_train[prot_col] == 0) & (y_pred == 0)] = 1/b
        elif fairness_criteria == 'get_calibration_fairness':
            b = math.sqrt((cfs[1][1, 1]*cfs[0][0, 1]) /
                          (cfs[0][1, 1]*cfs[1][0, 1]))
            b_ = math.sqrt((cfs[1][0, 0]*cfs[0][1, 0]) /
                           (cfs[0][0, 0]*cfs[1][1, 0]))
            b[(X_train[prot_col] == 1) & (
                y_train == 1) & (y_pred == 0)] = b
            b[(X_train[prot_col] == 0) & (
                y_train == 1) & (y_pred == 0)] = 1/b
            b[(X_train[prot_col] == 1) & (
                y_train == 0) & (y_pred == 1)] = b_
            b[(X_train[prot_col] == 0) & (
                y_train == 0) & (y_pred == 1)] = 1/b_
        elif fairness_criteria == 'get_differential_validity':
            b = math.sqrt((cfs[0][0, 0]*cfs[0][0, 1])/(cfs[0][1, 1]*cfs[0][1, 0]) +
                          (cfs[1][0, 0]*cfs[1][0, 1])/(cfs[1][1, 1]*cfs[1][1, 0]))
            b[(X_train[prot_col] == 1) & (y_pred != y_train)] = b
            b[(X_train[prot_col] == 0) & (y_pred != y_train)] = 1/b

        return b

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Perform a single boost according to the real multi-class SAMME.R
        algorithm or to the discrete SAMME algorithm and return the updated
        sample weights.

        Parameters
        ----------
        iboost : int
                The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples.

        y : array-like of shape (n_samples,)
                The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
                The current sample weights.

        random_state : RandomState instance
                The RandomState instance used if the base estimator accepts a
                `random_state` attribute.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
                The reweighted sample weights.
                If None then boosting has terminated early.

        estimator_weight : float
                The weight for the current boost.
                If None then boosting has terminated early.

        estimator_error : float
                The classification error for the current boost.
                If None then boosting has terminated early.
        """
        if self.algorithm == "SAMME.R":
            return self._boost_real(iboost, X, y, sample_weight, random_state)

        else:  # elif self.algorithm == "SAMME":
            return self._boost_discrete(iboost, X, y, sample_weight, random_state)

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = DecisionTreeClassifier(
            max_depth=1, random_state=random_state)
        self.estimators_.append(estimator)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(
            np.argmax(y_predict_proba, axis=1), axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(
            incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (
            -1.0
            * self.learning_rate
            * ((n_classes - 1.0) / n_classes)
            * xlogy(y_coding, y_predict_proba).sum(axis=1)
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(
                estimator_weight * ((sample_weight > 0) |
                                    (estimator_weight < 0))
            )

        return sample_weight, 1.0, estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = DecisionTreeClassifier(
            max_depth=1, random_state=random_state)
        self.estimators_.append(estimator)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(
            incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier "
                    "ensemble is worse than random, ensemble "
                    "can not be fit."
                )
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) +
            np.log(n_classes - 1.0)
        )
        confusion_matrices = []
        for col_name in [self.prot_col, self.ref_col]:
            tp = len(X[(y == 1) &
                       (y_predict == 1) &
                       (X[col_name] == 1)])
            fn = len(X[(y == 1) &
                       (y_predict == 0) &
                       (X[col_name] == 1)])
            fp = len(X[(y == 0) &
                       (y_predict == 1) &
                       (X[col_name] == 1)])
            tn = len(X[(y == 0) &
                       (y_predict == 0) &
                       (X[col_name] == 1)])
            confusion_matrices.append(np.array([[tp, fn], [fp, tn]]))
        b = self.fairyfy(self.prot_col, self.ref_col, X, y, y_predict,
                         self.fairness_criteria, confusion_matrices)

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight = b*np.exp(
                np.log(sample_weight)
                + estimator_weight * incorrect * (sample_weight > 0)
            )

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Sparse matrix can be CSC, CSR, COO,
                DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
                The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Sparse matrix can be CSC, CSR, COO,
                DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        score : ndarray of shape of (n_samples, k)
                The decision function of the input samples. The order of
                outputs is the same of that of the :term:`classes_` attribute.
                Binary classification is a special cases with ``k == 1``,
                otherwise ``k==n_classes``. For binary classification,
                values closer to -1 or 1 mean more like the first or second
                class in ``classes_``, respectively.
        """
        check_is_fitted(self)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        if self.algorithm == "SAMME.R":
            # The weights are all 1. for SAMME.R
            pred = sum(
                _samme_proba(estimator, n_classes, X) for estimator in self.estimators_
            )
        else:  # self.algorithm == "SAMME"
            pred = sum(
                (estimator.predict(X) == classes).T * w
                for estimator, w in zip(self.estimators_, self.estimator_weights_)
            )

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Sparse matrix can be CSC, CSR, COO,
                DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
                The target values.

        sample_weight : array-like of shape (n_samples,), default=None
                Sample weights. If None, the sample weights are initialized to
                1 / n_samples.

        Returns
        -------
        self : object
                Fitted estimator.
        """

        sample_weight = np.ones(len(X))
        sample_weight /= sample_weight.sum()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        # Initialization of the random number instance that will be used to
        # generate a seed at each iteration
        random_state = check_random_state(self.random_state)
        epsilon = np.finfo(sample_weight.dtype).eps

        zero_weight_mask = sample_weight == 0.0
        for iboost in range(self.n_estimators):
            # avoid extremely small sample weight, for details see issue #20320
            sample_weight = np.clip(sample_weight, a_min=epsilon, a_max=None)
            # do not clip sample weights that were exactly zero originally
            sample_weight[zero_weight_mask] = 0.0

            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight, random_state
            )

            # Early termination
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            if not np.isfinite(sample_weight_sum):
                warnings.warn(
                    "Sample weights have reached infinite values,"
                    f" at iteration {iboost}, causing overflow. "
                    "Iterations stopped. Try lowering the learning rate.",
                    stacklevel=2,
                )
                break

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self


if __name__ == '__main__':
    # fab = FABulous2(algorithm="SAMME", n_estimators=100, random_state=0)
    # fab.fit(X_train, y_train)
    # print(fab.score(X_test, y_test))

    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
    #                         algorithm = "SAMME", n_estimators = 100,
    #                        random_state = 0)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
    # print(clf.score(X_test, y_test))
    pass
