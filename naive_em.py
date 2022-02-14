"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    post_1 = np.zeros((n, K))

    for i in range(n):
        for j in range(K):
            post_1[i, j] = (1 / ((2 * np.pi * mixture.var[j]) ** (d / 2))) * np.exp(
                                 -1 / (2 * mixture.var[j]) * np.sum(((X[i, :] - mixture.mu[j, :]) * (X[i, :] - mixture.mu[j, :])))) * mixture.p[j]
        post[i, :] = post_1[i, :] / np.sum(post_1[i, :])

    the_sum = 0
    for i in range(n):
        sum_dm = 0
        for j in range(K):
            sum_dm += post_1[i, j]
        the_sum += np.log(sum_dm)
    return post, the_sum
    # raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    mu = np.zeros((K, d))
    var = np.zeros(K)
    p = np.zeros(K)

    for j in range(K):
        for i in range(n):
            mu[j, :] += X[i, :] * post[i, j] / np.sum(post[:, j])
        p[j] = 1 / n * np.sum(post[:, j])
        for i in range(n):
            var[j] += ((X[i, :] - mu[j, :])**2).sum() * post[i, j] / (d * np.sum(post[:, j]))

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
        mixture = mstep(X, post)

    return mixture, post, cost
    # raise NotImplementedError
