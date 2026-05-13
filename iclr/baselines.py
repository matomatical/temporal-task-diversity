"""Bayes-optimal baselines for in-context linear regression.

Implements:
- dmmse: Bayes-optimal predictor for discrete task distributions
- ridge: Bayes-optimal predictor for Gaussian prior over tasks
- compute_ridge_posterior: Analytic posterior parameters for ridge
- compute_dmmse_posterior: Discrete posterior weights over task pool
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import einops
from jaxtyping import Array, Float


@jax.jit
def dmmse(
    xs: Float[Array, "num_examples task_dim"],
    ys: Float[Array, "num_examples"],
    ws: Float[Array, "num_tasks task_dim"],
    noise_var: float,
) -> Float[Array, "num_examples"]:
    """Bayes-optimal predictions for discrete task distribution.

    For each prefix of the sequence, computes the posterior distribution over
    tasks using Boltzmann weighting, then predicts using the posterior mean.

    Args:
        xs: Input vectors [num_examples, task_dim]
        ys: Output values [num_examples]
        ws: Task weight vectors [num_tasks, task_dim]
        noise_var: Known variance of observation noise

    Returns:
        Predictions [num_examples] where pred[k] uses context xs[:k], ys[:k]
    """
    N, D = xs.shape

    # STAGE 1: compute the optimal solution vector for each prefix
    ws_hat = jnp.empty((N, D))

    # k = 0: no prefix, just use the mean task
    w_mean = ws.mean(axis=0)  # M D -> D
    ws_hat = ws_hat.at[0, :].set(w_mean)

    # k > 0: softmax-weighted average of each task by loss on prefix
    # (exclude final example as it is not used in any prefix)
    X = xs[: N - 1, :]  # -> N-1 D
    y = ys[: N - 1, jnp.newaxis]  # -> N-1 1
    # compute cumulative loss for each prefix
    y_hat = X @ ws.T  # N-1 D @ D M   -> N-1 M
    delta = y_hat - y  # N-1 M - N-1 1 -> N-1 M
    losses = (delta**2).cumsum(axis=0)  # N-1 M         -> N-1 M
    scores = -losses / (2 * noise_var)  # N-1 M / . .   -> N-1 M
    probs = jax.nn.softmax(scores, axis=-1)  # N-1 norm(M)   -> N-1 M(sum=1)
    ws_hat = ws_hat.at[1:, :].set(probs @ ws)  # N-1 M @ M D   -> N-1 D

    # STAGE 2: From these solution vectors, compute each prediction
    ys_pred = einops.einsum(xs, ws_hat, "N D, N D -> N")

    # that's all
    return ys_pred


@jax.jit
def dmmse_batch(
    xss: Float[Array, "batch_size num_examples task_dim"],
    yss: Float[Array, "batch_size num_examples"],
    ws: Float[Array, "num_tasks task_dim"],
    noise_var: float,
) -> Float[Array, "batch_size num_examples"]:
    """Batched dmmse via vmap."""
    return jax.vmap(dmmse, in_axes=(0, 0, None, None))(xss, yss, ws, noise_var)


@jax.jit
def ridge(
    xs: Float[Array, "num_examples task_dim"],
    ys: Float[Array, "num_examples"],
    noise_var: float,
) -> Float[Array, "num_examples"]:
    """Bayes-optimal predictions for Gaussian prior over tasks.

    Standard ridge regression estimator. For each prefix, solves the
    regularized normal equations.

    Args:
        xs: Input vectors [num_examples, task_dim]
        ys: Output values [num_examples]
        noise_var: Known variance of observation noise (serves as ridge penalty)

    Returns:
        Predictions [num_examples] where pred[k] uses context xs[:k], ys[:k]
    """
    N, D = xs.shape

    # STAGE 1: compute the optimal solution vector for each prefix
    ws_hat = jnp.empty((N, D))

    # k = 0: no prefix, just use the mean task
    w_mean = jnp.zeros((D,))
    ws_hat = ws_hat.at[0, :].set(w_mean)

    # k > 0: compute and then solve normal equations
    # (exclude final example as it is not used in any prefix)
    X = xs[: N - 1, :]  # -> N-1 D
    y = ys[: N - 1, jnp.newaxis]  # -> N-1 1
    # compute gram matrices for each n using outer product decomposition
    xTx = einops.einsum(X, X, "Nminus1 D1, Nminus1 D2 -> Nminus1 D1 D2")
    XTX = jnp.cumsum(xTx, axis=0)  # -> N-1 D D
    XTX_reg = XTX + noise_var * jnp.eye(D)  # N-1 D D + . D D -> K D D
    # same for RHS
    xTy = einops.einsum(X, y, "Nminus1 D, Nminus1 one -> Nminus1 D one")
    XTy = jnp.cumsum(xTy, axis=0)  # -> N-1 D 1
    # solve normal equations
    ws_hat = ws_hat.at[1:, :].set(
        jnp.linalg.solve(XTX_reg, XTy)[  # N-1 D D @ N-1 D 1 -> N-1 D 1
            :, :, 0
        ]  # -> N-1 D
    )

    # STAGE 2: From these solution vectors, compute each prediction
    ys_pred = einops.einsum(xs, ws_hat, "N D, N D -> N")

    # that's all!
    return ys_pred


@jax.jit
def ridge_batch(
    xss: Float[Array, "batch_size num_examples task_dim"],
    yss: Float[Array, "batch_size num_examples"],
    noise_var: float,
) -> Float[Array, "batch_size num_examples"]:
    """Batched ridge via vmap."""
    return jax.vmap(ridge, in_axes=(0, 0, None))(xss, yss, noise_var)


# # #
# Posterior computation


@jax.jit
def compute_ridge_posterior(
    xs: Float[Array, "num_examples task_dim"],
    ys: Float[Array, "num_examples"],
    noise_var: float,
    prior_var: float = 1.0,
) -> tuple[Float[Array, "task_dim"], Float[Array, "task_dim task_dim"]]:
    """Compute analytic ridge posterior parameters.

    Prior: w ~ N(0, prior_var * I)
    Likelihood: y | X, w ~ N(Xw, noise_var * I)
    Posterior: w | X, y ~ N(mu, Sigma)

    Args:
        xs: Input vectors [num_examples, task_dim]
        ys: Output values [num_examples]
        noise_var: Observation noise variance
        prior_var: Prior variance (default 1.0 for standard Gaussian prior)

    Returns:
        mu: Posterior mean [task_dim]
        Sigma: Posterior covariance [task_dim, task_dim]
    """
    D = xs.shape[1]

    XTX = xs.T @ xs  # D D
    XTy = xs.T @ ys  # D

    # precision matrix: X^T X / noise_var + I / prior_var
    precision = XTX / noise_var + jnp.eye(D) / prior_var
    Sigma = jnp.linalg.inv(precision)  # D D
    mu = Sigma @ (XTy / noise_var)  # D

    return mu, Sigma


@jax.jit
def compute_ridge_posterior_batch(
    xss: Float[Array, "batch_size num_examples task_dim"],
    yss: Float[Array, "batch_size num_examples"],
    noise_var: float,
    prior_var: float = 1.0,
) -> tuple[
    Float[Array, "batch_size task_dim"],
    Float[Array, "batch_size task_dim task_dim"],
]:
    """Batched compute_ridge_posterior via vmap."""
    return jax.vmap(compute_ridge_posterior, in_axes=(0, 0, None, None))(
        xss, yss, noise_var, prior_var
    )


@jax.jit
def compute_dmmse_posterior(
    xs: Float[Array, "num_examples task_dim"],
    ys: Float[Array, "num_examples"],
    ws: Float[Array, "num_tasks task_dim"],
    noise_var: float,
) -> Float[Array, "num_tasks"]:
    """Compute discrete posterior weights over task pool.

    Args:
        xs: Input vectors [num_examples, task_dim]
        ys: Output values [num_examples]
        ws: Task weight vectors [num_tasks, task_dim]
        noise_var: Observation noise variance

    Returns:
        weights: Posterior probability for each task [num_tasks]
    """
    # predictions for each task
    preds = xs @ ws.T  # N M
    residuals = ys[:, None] - preds  # N M

    # log likelihood (up to constant): -0.5 * sum of squared residuals / noise_var
    log_weights = -0.5 / noise_var * (residuals**2).sum(axis=0)  # M
    weights = jax.nn.softmax(log_weights)  # M

    return weights


@jax.jit
def compute_dmmse_posterior_batch(
    xss: Float[Array, "batch_size num_examples task_dim"],
    yss: Float[Array, "batch_size num_examples"],
    ws: Float[Array, "num_tasks task_dim"],
    noise_var: float,
) -> Float[Array, "batch_size num_tasks"]:
    """Batched compute_dmmse_posterior via vmap."""
    return jax.vmap(compute_dmmse_posterior, in_axes=(0, 0, None, None))(
        xss, yss, ws, noise_var
    )


