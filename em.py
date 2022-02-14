"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # initialize variables
    n, d = X.shape
    K, _ = mixture.mu.shape
    second_terms = np.zeros((n, K), dtype=np.float64)
    log_post = np.zeros((n, K), dtype=np.float64)
    ll_post = np.float64(0)

    # calculate 2nd term of f
    for i in range(n):
        x_dm = X[i, :]
        nonzero_index = np.nonzero(x_dm)
        nonzero_count = np.count_nonzero(x_dm)
        for j in range(K):
            mu_dm = mixture.mu[j, :]
            second_terms[i, j] = -np.log(2 * np.pi * mixture.var[j]) * (nonzero_count / 2) - 1 / (
                    2 * mixture.var[j]) * np.sum(((x_dm[nonzero_index] - mu_dm[nonzero_index]) ** 2), dtype=np.float64)

    # subtract 2nd term from f
    log_p = np.tile(np.log(mixture.p + 1e-16), (n, 1))
    f_dm = log_p + second_terms

    # calculate log-posterior
    for i in range(n):
        logsumexp_dm = logsumexp(f_dm[i])
        for j in range(K):
            log_post[i, j] = f_dm[i, j] - logsumexp_dm
        ll_post += logsumexp_dm

    # return exp of the log_post and the log-likelihood of posterior probs by adding all of the 2nd terms of f
    return np.exp(log_post), ll_post
    # raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    mu = np.copy(mixture.mu)
    var = np.array([min_variance] * K)
    p = np.zeros(K)
    delta = np.zeros((n, d))
    delta[X > 0] = 1

    for j in range(K):
        for coord in range(d):
            if np.log(np.sum(post[:, j] * delta[:, coord])) >= 0:
                mu[j, coord] = np.exp(np.log(np.sum(post[:, j] * delta[:, coord] * X[:, coord])) - np.log(
                    np.sum(post[:, j] * delta[:, coord])))

        ssq = np.zeros(n)
        nonzero_counts = np.zeros(n)
        for i in range(n):
            x_dm = X[i, :]
            nonzero_idx = np.nonzero(x_dm)
            nonzero_counts[i] = np.count_nonzero(x_dm)
            mu_dm = np.copy(mu[j, :])
            ssq[i] = ((x_dm[nonzero_idx] - mu_dm[nonzero_idx])**2).sum()

        var_dm = np.exp(np.log(np.sum(post[:, j] * ssq)) - np.log(np.sum(nonzero_counts * post[:, j])))
        if var_dm > min_variance:
            var[j] = var_dm

        p[j] = 1 / n * np.sum(post[:, j])

    return GaussianMixture(mu, var, p)
    # raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while (prev_cost is None or cost - prev_cost > 1e-6 * np.abs(cost)):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, cost
    # raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X_copy = X.copy()
    n, d = X.shape
    K, _ = mixture.mu.shape
    assigned_cluster = np.array([0] * n)

    second_terms = np.zeros((n, K), dtype=np.float64)
    log_post = np.zeros((n, K), dtype=np.float64)

    # calculate 2nd term of f
    for i in range(n):
        x_dm = X[i, :]
        nonzero_index = np.nonzero(x_dm)
        nonzero_count = np.count_nonzero(x_dm)
        for j in range(K):
            mu_dm = mixture.mu[j, :]
            second_terms[i, j] = -np.log(2 * np.pi * mixture.var[j]) * (nonzero_count / 2) - 1 / (
                    2 * mixture.var[j]) * np.sum(((x_dm[nonzero_index] - mu_dm[nonzero_index]) ** 2), dtype=np.float64)

    # subtract 2nd term from f
    log_p = np.tile(np.log(mixture.p + 1e-16), (n, 1))
    f_dm = log_p + second_terms

    # calculate log-posterior
    for i in range(n):
        logsumexp_dm = logsumexp(f_dm[i])
        for j in range(K):
            log_post[i, j] = f_dm[i, j] - logsumexp_dm

    # posterior probabilities
    post_prob = np.exp(log_post)

    # filling the matrix
    for i in range(n):
        for coord in range(d):
            if X_copy[i, coord] == 0:
                X_copy[i, coord] = np.sum(post_prob[i, :] * mixture.mu[:, coord])
    return X_copy
    # raise NotImplementedError
