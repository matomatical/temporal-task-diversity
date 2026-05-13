from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .generation import XGenerator, generate_batch
from .setting import InContextRegressionTransformer


def _ols_beta(
    xs: Float[Array, "num_examples task_dim"],
    ys: Float[Array, "num_examples"],
) -> Float[Array, "task_dim"]:
    """Compute OLS beta: (X^T X)^{-1} X^T y."""
    XTX = xs.T @ xs
    XTy = xs.T @ ys
    return jnp.linalg.solve(XTX, XTy)


@functools.partial(
    jax.jit,
    static_argnames=[
        "num_samples",
        "num_steps",
        "x_generator",
        "return_ys",
        "print_progress",
    ],
)
def montecarlo_task_estimate(
    model: InContextRegressionTransformer,
    key: PRNGKeyArray,
    *,
    num_samples: int,
    num_steps: int,
    noise_var: float,
    x_generator: XGenerator = jax.random.normal,
    return_ys: bool = False,
    print_progress: bool = False,
) -> (
    Float[Array, "num_samples task_dim"]
    | tuple[
        Float[Array, "num_samples task_dim"],
        Float[Array, "num_samples num_steps"],
    ]
):
    """Compute unconditional task-vector estimates via predictive Monte Carlo.

    Generates sequences from the model with no prompt conditioning, then fits OLS
    to each generated sequence to obtain a task estimate.

    Args:
        model: Trained InContextRegressionTransformer
        key: PRNG key for sampling
        num_samples: Number of sequences to generate
        num_steps: Length of each generated sequence
        noise_var: Noise variance for gaussian/point heads
        x_generator: Function to generate random x values (default: jax.random.normal)
        return_ys: If True, also return generated y values

    Returns:
        task_estimates: OLS coefficients [num_samples, task_dim]
        If return_ys=True, returns (task_estimates, ys) where ys has shape
        [num_samples, num_steps]
    """
    xs_gen, ys_gen = generate_batch(
        model,
        key,
        num_generations=num_samples,
        num_examples=num_steps,
        noise_var=noise_var,
        x_generator=x_generator,
        print_progress=print_progress,
    )

    task_estimates = jax.vmap(_ols_beta)(xs_gen, ys_gen)

    if return_ys:
        return task_estimates, ys_gen
    return task_estimates
