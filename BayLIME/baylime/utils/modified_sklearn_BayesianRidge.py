# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:26:54 2020

@author: Xingyu Zhao
"""

"""
Various bayesian regression
XZ did a new function to replace the sklearn BayesianRidge with 
BayesianRidge_inf_prior. Basically it does not allow BayesianRidge
to do automatic model selection for finding optimum alpha and lambda
to let ``the data speak for themselves'', rather we specify them manually
as informative priors from humans.

"""



from math import log
import numpy as np
from scipy import linalg


from ._base import LinearModel
from ..base import RegressorMixin
from ..utils.extmath import fast_logdet
from ..utils import check_X_y
from scipy.linalg import pinvh
from scipy import sparse

from ..utils.validation import _check_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y
from sklearn.utils.extmath import safe_sparse_dot

###############################################################################
# BayesianRidge regression

class BayesianRidge_inf_prior(RegressorMixin, LinearModel):

    def __init__(self, n_iter=0, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6,
                 lambda_1=1.e-6, lambda_2=1.e-6, alpha_init=None,
                 lambda_init=None, compute_score=False, fit_intercept=True,
                 normalize=False, copy_X=True, verbose=False):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.verbose = verbose
        
    def _preprocess_data(self, X, y, fit_intercept, normalize=False, copy=True, sample_weight=None):
        if copy:
            X = X.copy()
            y = y.copy()

        # Compute offsets
        if fit_intercept:
            if sample_weight is not None:
                X_offset = np.average(X, axis=0, weights=sample_weight)
                y_offset = np.average(y, axis=0, weights=sample_weight)
            else:
                X_offset = X.mean(axis=0)
                y_offset = y.mean(axis=0)
            X -= X_offset
            y -= y_offset
        else:
            X_offset = np.zeros(X.shape[1], dtype=X.dtype)
            y_offset = 0.0

        # Compute scales
        if normalize:
            if sample_weight is not None:
                X_scale = np.sqrt(np.average(X ** 2, axis=0, weights=sample_weight))
            else:
                X_scale = np.linalg.norm(X, axis=0)
            # Avoid division by zero
            X_scale[X_scale == 0] = 1.0
            X /= X_scale
        else:
            X_scale = np.ones(X.shape[1], dtype=X.dtype)

        return X, y, X_offset, y_offset, X_scale

        
    def _rescale_data(self, X, y, sample_weight):
        """Rescale data sample-wise by square root of sample_weight."""
        n_samples = X.shape[0]
        sample_weight = np.asarray(sample_weight)
        if sample_weight.ndim == 0:
            sample_weight = np.full(n_samples, sample_weight,
                                    dtype=sample_weight.dtype)
        sample_weight = np.sqrt(sample_weight)
        sw_matrix = sparse.dia_matrix((sample_weight, 0),
                                      shape=(n_samples, n_samples))
        X = safe_sparse_dot(sw_matrix, X)
        y = sample_weight * y
        return X, y


    def fit(self, X, y, sample_weight=None):
        if self.n_iter != 0:
            raise ValueError('n_iter can only be 0. Got {!r}.'.format(self.n_iter))

        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)


               # Preprocess data
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X, sample_weight=sample_weight
        )

        # Store fitted preprocessing parameters
        self.X_offset_ = X_offset
        self.y_offset_ = y_offset
        self.X_scale_ = X_scale

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = self._rescale_data(X, y, sample_weight)

        n_samples, n_features = X.shape


        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero
        alpha_ = self.alpha_init
        lambda_ = self.lambda_init
        if alpha_ is None:
            alpha_ = 1. / (np.var(y) + eps)
        if lambda_ is None:
            lambda_ = 1.
        
        #XZ:not using these parameters, since we are not doing model selection
        verbose = self.verbose
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        U, S, Vh = linalg.svd(X, full_matrices=False)
        eigen_vals_ = S ** 2

        # Convergence loop of the bayesian ridge regression
        

        
        # for iter_ in range(self.n_iter):

        #     # update posterior mean coef_ based on alpha_ and lambda_ and
        #     # compute corresponding rmse
        #     coef_, rmse_ = self._update_coef_(X, y, n_samples, n_features,
        #                                       XT_y, U, Vh, eigen_vals_,
        #                                       alpha_, lambda_)
        #     if self.compute_score:
        #         # compute the log marginal likelihood
        #         s = self._log_marginal_likelihood(n_samples, n_features,
        #                                           eigen_vals_,
        #                                           alpha_, lambda_,
        #                                           coef_, rmse_)
        #         self.scores_.append(s)

        #     # Update alpha and lambda according to (MacKay, 1992)
        #     gamma_ = np.sum((alpha_ * eigen_vals_) /
        #                     (lambda_ + alpha_ * eigen_vals_))
        #     lambda_ = ((gamma_ + 2 * lambda_1) /
        #                 (np.sum(coef_ ** 2) + 2 * lambda_2))
        #     alpha_ = ((n_samples - gamma_ + 2 * alpha_1) /
        #               (rmse_ + 2 * alpha_2))

        #     # Check for convergence
        #     if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
        #         if verbose:
        #             print("Convergence after ", str(iter_), " iterations")
        #         break
        #     coef_old_ = np.copy(coef_)

        # self.n_iter_ = iter_ + 1
        
        

        # return regularization parameters and corresponding posterior mean,
        # log marginal likelihood and posterior covariance
        
        #XZ: just use the inital alpha and lambda do the update..
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.coef_, rmse_ = self._update_coef_(X, y, n_samples, n_features,
                                               XT_y, U, Vh, eigen_vals_,
                                               alpha_, lambda_)
        
        coef_=self.coef_                               
        if self.compute_score:
            # compute the log marginal likelihood
            s = self._log_marginal_likelihood(n_samples, n_features,
                                              eigen_vals_,
                                              alpha_, lambda_,
                                              coef_, rmse_)
            self.scores_.append(s)
            self.scores_ = np.array(self.scores_)
        

        # posterior covariance is given by 1/alpha_ * scaled_sigma_
        scaled_sigma_ = np.dot(Vh.T,
                               Vh / (eigen_vals_ +
                                     lambda_ / alpha_)[:, np.newaxis])
        self.sigma_ = (1. / alpha_) * scaled_sigma_

        self._set_intercept(self.X_offset_, self.y_offset_, self.X_scale_)


        return self

    def predict(self, X, return_std=False):
        """Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y_mean : array-like of shape (n_samples,)
            Mean of predictive distribution of query points.

        y_std : array-like of shape (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
        y_mean = self._decision_function(X)
        if return_std is False:
            return y_mean
        else:
            if self.normalize:
                X = (X - self.X_offset_) / self.X_scale_
            sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + (1. / self.alpha_))
            return y_mean, y_std

    def _update_coef_(self, X, y, n_samples, n_features, XT_y, U, Vh,
                      eigen_vals_, alpha_, lambda_):
        """Update posterior mean and compute corresponding rmse.

        Posterior mean is given by coef_ = scaled_sigma_ * X.T * y where
        scaled_sigma_ = (lambda_/alpha_ * np.eye(n_features)
                         + np.dot(X.T, X))^-1
        """

        if n_samples > n_features:
            coef_ = np.dot(Vh.T,
                           Vh / (eigen_vals_ +
                                 lambda_ / alpha_)[:, np.newaxis])
            coef_ = np.dot(coef_, XT_y)
        else:
            coef_ = np.dot(X.T, np.dot(
                U / (eigen_vals_ + lambda_ / alpha_)[None, :], U.T))
            coef_ = np.dot(coef_, y)

        rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)

        return coef_, rmse_

    def _log_marginal_likelihood(self, n_samples, n_features, eigen_vals,
                                 alpha_, lambda_, coef, rmse):
        """Log marginal likelihood."""
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # compute the log of the determinant of the posterior covariance.
        # posterior covariance is given by
        # sigma = (lambda_ * np.eye(n_features) + alpha_ * np.dot(X.T, X))^-1
        if n_samples > n_features:
            logdet_sigma = - np.sum(np.log(lambda_ + alpha_ * eigen_vals))
        else:
            logdet_sigma = np.full(n_features, lambda_,
                                   dtype=np.array(lambda_).dtype)
            logdet_sigma[:n_samples] += alpha_ * eigen_vals
            logdet_sigma = - np.sum(np.log(logdet_sigma))

        score = lambda_1 * log(lambda_) - lambda_2 * lambda_
        score += alpha_1 * log(alpha_) - alpha_2 * alpha_
        score += 0.5 * (n_features * log(lambda_) +
                        n_samples * log(alpha_) -
                        alpha_ * rmse -
                        lambda_ * np.sum(coef ** 2) +
                        logdet_sigma -
                        n_samples * log(2 * np.pi))

        return score









###############################################################################
# BayesianRidge regression modified by XZ

class BayesianRidge_inf_prior_fit_alpha(RegressorMixin, LinearModel):

    def __init__(self, n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6,
                 lambda_1=1.e-6, lambda_2=1.e-6, alpha_init=None,
                 lambda_init=None, compute_score=False, fit_intercept=True,
                 normalize=False, copy_X=True, verbose=False):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.verbose = verbose
        
    def _preprocess_data(self, X, y, fit_intercept, normalize=False, copy=True, sample_weight=None):
        if copy:
            X = X.copy()
            y = y.copy()

        # Compute offsets
        if fit_intercept:
            if sample_weight is not None:
                X_offset = np.average(X, axis=0, weights=sample_weight)
                y_offset = np.average(y, axis=0, weights=sample_weight)
            else:
                X_offset = X.mean(axis=0)
                y_offset = y.mean(axis=0)
            X -= X_offset
            y -= y_offset
        else:
            X_offset = np.zeros(X.shape[1], dtype=X.dtype)
            y_offset = 0.0

        # Compute scales
        if normalize:
            if sample_weight is not None:
                X_scale = np.sqrt(np.average(X ** 2, axis=0, weights=sample_weight))
            else:
                X_scale = np.linalg.norm(X, axis=0)
            # Avoid division by zero
            X_scale[X_scale == 0] = 1.0
            X /= X_scale
        else:
            X_scale = np.ones(X.shape[1], dtype=X.dtype)

        return X, y, X_offset, y_offset, X_scale

        
    def _rescale_data(self, X, y, sample_weight):
        """Rescale data sample-wise by square root of sample_weight."""
        n_samples = X.shape[0]
        sample_weight = np.asarray(sample_weight)
        if sample_weight.ndim == 0:
            sample_weight = np.full(n_samples, sample_weight,
                                    dtype=sample_weight.dtype)
        sample_weight = np.sqrt(sample_weight)
        sw_matrix = sparse.dia_matrix((sample_weight, 0),
                                      shape=(n_samples, n_samples))
        X = safe_sparse_dot(sw_matrix, X)
        y = sample_weight * y
        return X, y

        
    def fit(self, X, y, sample_weight=None):
        """Fit the model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary
        

        sample_weight : ndarray of shape (n_samples,), default=None
            Individual weights for each sample

            .. versionadded:: 0.20
               parameter *sample_weight* support to BayesianRidge.

        Returns
        -------
        self : returns an instance of self.
        """

        if self.n_iter < 1:
            raise ValueError('n_iter should be greater than or equal to 1.'
                             ' Got {!r}.'.format(self.n_iter))

        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        
        # #to be deleted by XZ
        # Z=np.dot(X.T, X)
        # print(Z)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        # Preprocess data
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X, sample_weight=sample_weight
        )

        # Store fitted preprocessing parameters
        self.X_offset_ = X_offset
        self.y_offset_ = y_offset
        self.X_scale_ = X_scale

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = self._rescale_data(X, y, sample_weight)

        n_samples, n_features = X.shape

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero
        alpha_ = self.alpha_init
        lambda_ = self.lambda_init
        if alpha_ is None:
            alpha_ = 1. / (np.var(y) + eps)
        if lambda_ is None:
            lambda_ = 1.

        verbose = self.verbose
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        U, S, Vh = linalg.svd(X, full_matrices=False)
        eigen_vals_ = S ** 2

        # Convergence loop of the bayesian ridge regression
        

        
        for iter_ in range(self.n_iter):

            # update posterior mean coef_ based on alpha_ and lambda_ and
            # compute corresponding rmse
            coef_, rmse_ = self._update_coef_(X, y, n_samples, n_features,
                                              XT_y, U, Vh, eigen_vals_,
                                              alpha_, lambda_)
            if self.compute_score:
                # compute the log marginal likelihood
                s = self._log_marginal_likelihood(n_samples, n_features,
                                                  eigen_vals_,
                                                  alpha_, lambda_,
                                                  coef_, rmse_)
                self.scores_.append(s)

            # Update alpha and lambda according to (MacKay, 1992)
            gamma_ = np.sum((alpha_ * eigen_vals_) /
                            (lambda_ + alpha_ * eigen_vals_))
            
            lambda_=lambda_
            # lambda_ = ((gamma_ + 2 * lambda_1) /
            #             (np.sum(coef_ ** 2) + 2 * lambda_2))
            alpha_ = ((n_samples - gamma_ + 2 * alpha_1) /
                      (rmse_ + 2 * alpha_2))

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            coef_old_ = np.copy(coef_)

        self.n_iter_ = iter_ + 1
        
        

        # return regularization parameters and corresponding posterior mean,
        # log marginal likelihood and posterior covariance
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.coef_, rmse_ = self._update_coef_(X, y, n_samples, n_features,
                                               XT_y, U, Vh, eigen_vals_,
                                               alpha_, lambda_)
        
        coef_=self.coef_                               
        if self.compute_score:
            # compute the log marginal likelihood
            s = self._log_marginal_likelihood(n_samples, n_features,
                                              eigen_vals_,
                                              alpha_, lambda_,
                                              coef_, rmse_)
            self.scores_.append(s)
            self.scores_ = np.array(self.scores_)
        
        
        
        # #XZ added, to be deleted
        # Z=np.dot(X.T, X)
        # Z=lambda_/alpha_ * np.eye(n_features) + Z
        # Z=np.linalg.inv(Z)
        
        
        # posterior covariance is given by 1/alpha_ * scaled_sigma_
        scaled_sigma_ = np.dot(Vh.T,
                               Vh / (eigen_vals_ +
                                     lambda_ / alpha_)[:, np.newaxis])
        self.sigma_ = (1. / alpha_) * scaled_sigma_

        self._set_intercept(self.X_offset_, self.y_offset_, self.X_scale_)

        return self

    def predict(self, X, return_std=False):
        """Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y_mean : array-like of shape (n_samples,)
            Mean of predictive distribution of query points.

        y_std : array-like of shape (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
        y_mean = self._decision_function(X)
        if return_std is False:
            return y_mean
        else:
            if self.normalize:
                X = (X - self.X_offset_) / self.X_scale_
            sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + (1. / self.alpha_))
            return y_mean, y_std

    def _update_coef_(self, X, y, n_samples, n_features, XT_y, U, Vh,
                      eigen_vals_, alpha_, lambda_):
        """Update posterior mean and compute corresponding rmse.

        Posterior mean is given by coef_ = scaled_sigma_ * X.T * y where
        scaled_sigma_ = (lambda_/alpha_ * np.eye(n_features)
                         + np.dot(X.T, X))^-1
        """
        
        
        if n_samples > n_features:
            coef_ = np.dot(Vh.T,
                           Vh / (eigen_vals_ +
                                 lambda_ / alpha_)[:, np.newaxis])
            coef_ = np.dot(coef_, XT_y)
        else:
            coef_ = np.dot(X.T, np.dot(
                U / (eigen_vals_ + lambda_ / alpha_)[None, :], U.T))
            coef_ = np.dot(coef_, y)

        rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)

        return coef_, rmse_

    def _log_marginal_likelihood(self, n_samples, n_features, eigen_vals,
                                 alpha_, lambda_, coef, rmse):
        """Log marginal likelihood."""
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # compute the log of the determinant of the posterior covariance.
        # posterior covariance is given by
        # sigma = (lambda_ * np.eye(n_features) + alpha_ * np.dot(X.T, X))^-1
        if n_samples > n_features:
            logdet_sigma = - np.sum(np.log(lambda_ + alpha_ * eigen_vals))
        else:
            logdet_sigma = np.full(n_features, lambda_,
                                   dtype=np.array(lambda_).dtype)
            logdet_sigma[:n_samples] += alpha_ * eigen_vals
            logdet_sigma = - np.sum(np.log(logdet_sigma))

        score = lambda_1 * log(lambda_) - lambda_2 * lambda_
        score += alpha_1 * log(alpha_) - alpha_2 * alpha_
        score += 0.5 * (n_features * log(lambda_) +
                        n_samples * log(alpha_) -
                        alpha_ * rmse -
                        lambda_ * np.sum(coef ** 2) +
                        logdet_sigma -
                        n_samples * log(2 * np.pi))

        return score
