"""Train an in-context regression transformer at one configuration.

Stationary by default. Set --mala-step-size for MALA random-walk non-stationarity or --num-resamples for Dirichlet resampling non-stationarity (mutually exclusive). With --compute-energy-distance, additionally records the energy distance between the transformer's implicit prior and the dMMSE / ridge priors at each evaluation step.

Writes per-step metrics to <output_root>/<run_name>.json and an orbax checkpoint to <output_root>/checkpoints/<run_name>/. Re-invoking with the same --run-name resumes from the latest saved checkpoint, pass --force-restart to discard this checkpoint and start again with the same run_name.
"""

from __future__ import annotations

import datetime
import functools
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import strux
import tyro
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm

from iclr import (
    DiscreteTaskDistribution,
    GaussianTaskDistribution,
    InContextRegressionTransformer,
    RegressionSequenceDistribution,
    TrainState,
    dmmse_batch,
    gaussian_nll,
    mog_nll,
    ridge_batch,
    montecarlo_task_estimate,
)



def materialize_metrics(metrics: dict) -> dict:
    """Convert JAX arrays to Python floats with single sync point."""
    result = {}
    synced = False
    for k, v in metrics.items():
        if hasattr(v, "item"):
            if not synced:
                jax.block_until_ready(v)
                synced = True
            result[k] = float(v)
        else:
            result[k] = v
    return result


def _compute_eval_metrics_point(
    model: "InContextRegressionTransformer",
    xs: Float[Array, "batch n task_dim"],
    ys: Float[Array, "batch n"],
    xs_batch: Float[Array, "batch_train n task_dim"],
    ys_batch: Float[Array, "batch_train n"],
    train_tasks: Float[Array, "num_tasks task_dim"],
    test_tasks: Float[Array, "num_tasks task_dim"],
    noise_var: float,
    task_dim: int,
    delta_prompt_length: int,
    delta_tail_start: int | None,
) -> dict[str, Float[Array, ""]]:
    """Compute eval metrics for point head."""
    ys_test_pred = model.forward_point_batch(xs, ys)
    test_error = ((ys_test_pred - ys) ** 2).mean()

    ys_test_pred_ridge = ridge_batch(xs, ys, noise_var)
    test_error_ridge = ((ys - ys_test_pred_ridge) ** 2).mean()

    ys_test_pred_dmmse = dmmse_batch(xs, ys, test_tasks, noise_var)
    test_error_dmmse = ((ys - ys_test_pred_dmmse) ** 2).mean()

    ys_train_pred = model.forward_point_batch(xs_batch, ys_batch)

    ys_train_pred_ridge = ridge_batch(xs_batch, ys_batch, noise_var)
    train_error_ridge = ((ys_batch - ys_train_pred_ridge) ** 2).mean()

    ys_train_pred_dmmse = dmmse_batch(xs_batch, ys_batch, train_tasks, noise_var)
    train_error_dmmse = ((ys_batch - ys_train_pred_dmmse) ** 2).mean()

    metrics = {
        "test_error": test_error,
        "test_error_ridge": test_error_ridge,
        "test_error_dmmse": test_error_dmmse,
        "train_error_ridge": train_error_ridge,
        "train_error_dmmse": train_error_dmmse,
        "test_delta_ridge": (
            (
                ys_test_pred[:, :delta_prompt_length]
                - ys_test_pred_ridge[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
        "test_delta_dmmse": (
            (
                ys_test_pred[:, :delta_prompt_length]
                - ys_test_pred_dmmse[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
        "train_delta_ridge": (
            (
                ys_train_pred[:, :delta_prompt_length]
                - ys_train_pred_ridge[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
        "train_delta_dmmse": (
            (
                ys_train_pred[:, :delta_prompt_length]
                - ys_train_pred_dmmse[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
    }
    if delta_tail_start is not None:
        metrics["test_delta_tail_ridge"] = (
            (
                ys_test_pred[:, delta_tail_start:]
                - ys_test_pred_ridge[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
        metrics["test_delta_tail_dmmse"] = (
            (
                ys_test_pred[:, delta_tail_start:]
                - ys_test_pred_dmmse[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
        metrics["train_delta_tail_ridge"] = (
            (
                ys_train_pred[:, delta_tail_start:]
                - ys_train_pred_ridge[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
        metrics["train_delta_tail_dmmse"] = (
            (
                ys_train_pred[:, delta_tail_start:]
                - ys_train_pred_dmmse[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
    return metrics


def _compute_eval_metrics_gaussian(
    model: "InContextRegressionTransformer",
    xs: Float[Array, "batch n task_dim"],
    ys: Float[Array, "batch n"],
    xs_batch: Float[Array, "batch_train n task_dim"],
    ys_batch: Float[Array, "batch_train n"],
    train_tasks: Float[Array, "num_tasks task_dim"],
    test_tasks: Float[Array, "num_tasks task_dim"],
    noise_var: float,
    task_dim: int,
    delta_prompt_length: int,
    delta_tail_start: int | None,
) -> dict[str, Float[Array, ""]]:
    """Compute eval metrics for gaussian head."""
    ys_test_pred, ys_test_var = model.forward_gaussian_batch(
        xs, ys, noise_var=noise_var
    )
    test_error = ((ys_test_pred - ys) ** 2).mean()
    test_nll = gaussian_nll(ys, ys_test_pred, ys_test_var).mean()

    ys_test_pred_ridge = ridge_batch(xs, ys, noise_var)
    test_error_ridge = ((ys - ys_test_pred_ridge) ** 2).mean()

    ys_test_pred_dmmse = dmmse_batch(xs, ys, test_tasks, noise_var)
    test_error_dmmse = ((ys - ys_test_pred_dmmse) ** 2).mean()

    ys_train_pred, _ = model.forward_gaussian_batch(
        xs_batch, ys_batch, noise_var=noise_var
    )

    ys_train_pred_ridge = ridge_batch(xs_batch, ys_batch, noise_var)
    train_error_ridge = ((ys_batch - ys_train_pred_ridge) ** 2).mean()

    ys_train_pred_dmmse = dmmse_batch(xs_batch, ys_batch, train_tasks, noise_var)
    train_error_dmmse = ((ys_batch - ys_train_pred_dmmse) ** 2).mean()

    metrics = {
        "test_error": test_error,
        "test_nll": test_nll,
        "test_error_ridge": test_error_ridge,
        "test_error_dmmse": test_error_dmmse,
        "train_error_ridge": train_error_ridge,
        "train_error_dmmse": train_error_dmmse,
        "test_delta_ridge": (
            (
                ys_test_pred[:, :delta_prompt_length]
                - ys_test_pred_ridge[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
        "test_delta_dmmse": (
            (
                ys_test_pred[:, :delta_prompt_length]
                - ys_test_pred_dmmse[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
        "train_delta_ridge": (
            (
                ys_train_pred[:, :delta_prompt_length]
                - ys_train_pred_ridge[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
        "train_delta_dmmse": (
            (
                ys_train_pred[:, :delta_prompt_length]
                - ys_train_pred_dmmse[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
    }
    if delta_tail_start is not None:
        metrics["test_delta_tail_ridge"] = (
            (
                ys_test_pred[:, delta_tail_start:]
                - ys_test_pred_ridge[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
        metrics["test_delta_tail_dmmse"] = (
            (
                ys_test_pred[:, delta_tail_start:]
                - ys_test_pred_dmmse[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
        metrics["train_delta_tail_ridge"] = (
            (
                ys_train_pred[:, delta_tail_start:]
                - ys_train_pred_ridge[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
        metrics["train_delta_tail_dmmse"] = (
            (
                ys_train_pred[:, delta_tail_start:]
                - ys_train_pred_dmmse[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
    return metrics


def _compute_eval_metrics_mog(
    model: "InContextRegressionTransformer",
    xs: Float[Array, "batch n task_dim"],
    ys: Float[Array, "batch n"],
    xs_batch: Float[Array, "batch_train n task_dim"],
    ys_batch: Float[Array, "batch_train n"],
    train_tasks: Float[Array, "num_tasks task_dim"],
    test_tasks: Float[Array, "num_tasks task_dim"],
    noise_var: float,
    task_dim: int,
    delta_prompt_length: int,
    delta_tail_start: int | None,
) -> dict[str, Float[Array, ""]]:
    """Compute eval metrics for MoG head."""
    logits, means, vars = model.forward_mog_batch(xs, ys)
    pi = jax.nn.softmax(logits, axis=-1)
    ys_test_pred = (pi * means).sum(axis=-1)
    test_error = ((ys_test_pred - ys) ** 2).mean()
    test_nll = mog_nll(ys, logits, means, vars).mean()

    ys_test_pred_ridge = ridge_batch(xs, ys, noise_var)
    test_error_ridge = ((ys - ys_test_pred_ridge) ** 2).mean()

    ys_test_pred_dmmse = dmmse_batch(xs, ys, test_tasks, noise_var)
    test_error_dmmse = ((ys - ys_test_pred_dmmse) ** 2).mean()

    train_logits, train_means, _ = model.forward_mog_batch(xs_batch, ys_batch)
    train_pi = jax.nn.softmax(train_logits, axis=-1)
    ys_train_pred = (train_pi * train_means).sum(axis=-1)

    ys_train_pred_ridge = ridge_batch(xs_batch, ys_batch, noise_var)
    train_error_ridge = ((ys_batch - ys_train_pred_ridge) ** 2).mean()

    ys_train_pred_dmmse = dmmse_batch(xs_batch, ys_batch, train_tasks, noise_var)
    train_error_dmmse = ((ys_batch - ys_train_pred_dmmse) ** 2).mean()

    metrics = {
        "test_error": test_error,
        "test_nll": test_nll,
        "test_error_ridge": test_error_ridge,
        "test_error_dmmse": test_error_dmmse,
        "train_error_ridge": train_error_ridge,
        "train_error_dmmse": train_error_dmmse,
        "test_delta_ridge": (
            (
                ys_test_pred[:, :delta_prompt_length]
                - ys_test_pred_ridge[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
        "test_delta_dmmse": (
            (
                ys_test_pred[:, :delta_prompt_length]
                - ys_test_pred_dmmse[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
        "train_delta_ridge": (
            (
                ys_train_pred[:, :delta_prompt_length]
                - ys_train_pred_ridge[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
        "train_delta_dmmse": (
            (
                ys_train_pred[:, :delta_prompt_length]
                - ys_train_pred_dmmse[:, :delta_prompt_length]
            )
            ** 2
        ).mean()
        / task_dim,
    }
    if delta_tail_start is not None:
        metrics["test_delta_tail_ridge"] = (
            (
                ys_test_pred[:, delta_tail_start:]
                - ys_test_pred_ridge[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
        metrics["test_delta_tail_dmmse"] = (
            (
                ys_test_pred[:, delta_tail_start:]
                - ys_test_pred_dmmse[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
        metrics["train_delta_tail_ridge"] = (
            (
                ys_train_pred[:, delta_tail_start:]
                - ys_train_pred_ridge[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
        metrics["train_delta_tail_dmmse"] = (
            (
                ys_train_pred[:, delta_tail_start:]
                - ys_train_pred_dmmse[:, delta_tail_start:]
            )
            ** 2
        ).mean() / task_dim
    return metrics


_compute_eval_metrics_point_jit = jax.jit(
    _compute_eval_metrics_point,
    static_argnames=["task_dim", "delta_prompt_length", "delta_tail_start"],
)
_compute_eval_metrics_gaussian_jit = jax.jit(
    _compute_eval_metrics_gaussian,
    static_argnames=["task_dim", "delta_prompt_length", "delta_tail_start"],
)
_compute_eval_metrics_mog_jit = jax.jit(
    _compute_eval_metrics_mog,
    static_argnames=["task_dim", "delta_prompt_length", "delta_tail_start"],
)


@jax.jit
def energy_distance_multidim(
    samples_a: Float[Array, "n d"],
    samples_b: Float[Array, "m d"],
) -> Float[Array, ""]:
    """Compute energy distance between two multivariate sample sets.

    ED = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]

    Args:
        samples_a: Samples from first distribution [n, d]
        samples_b: Samples from second distribution [m, d]

    Returns:
        Energy distance (scalar)
    """
    n, m = samples_a.shape[0], samples_b.shape[0]

    # E[||X-Y||] - cross term
    cross_diffs = samples_a[:, None, :] - samples_b[None, :, :]
    cross_norms = jnp.linalg.norm(cross_diffs, axis=2)
    e_xy = jnp.mean(cross_norms)

    # E[||X-X'||] - within first distribution
    a_diffs = samples_a[:, None, :] - samples_a[None, :, :]
    a_norms = jnp.linalg.norm(a_diffs, axis=2)
    # exclude diagonal (self-distances = 0)
    e_xx = jnp.where(n > 1, jnp.sum(a_norms) / (n * (n - 1)), 0.0)

    # E[||Y-Y'||] - within second distribution
    b_diffs = samples_b[:, None, :] - samples_b[None, :, :]
    b_norms = jnp.linalg.norm(b_diffs, axis=2)
    e_yy = jnp.where(m > 1, jnp.sum(b_norms) / (m * (m - 1)), 0.0)

    return 2 * e_xy - e_xx - e_yy


def _dirichlet_alpha1(key: PRNGKeyArray, shape: tuple[int, int]) -> Array:
    """Sample Dirichlet(alpha=1) via normalized i.i.d. exponential(1)."""
    u = jax.random.uniform(key, shape=shape, minval=1e-6, maxval=1.0)
    e = -jnp.log(u)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def _proportions_to_integer_lengths(p: np.ndarray, total: int) -> np.ndarray:
    """Convert proportions (sum=1) to integer segment lengths summing to total."""
    raw = p * float(total)
    base = np.floor(raw).astype(np.int32)
    remainder = int(total - base.sum())
    if remainder > 0:
        frac = raw - base
        idx = np.argsort(-frac)
        base[idx[:remainder]] += 1
    if base.sum() != total:
        diff = int(total - base.sum())
        base[0] += diff
    return base


@strux.struct
class DirichletResetSchedule:
    tasks: Float[Array, "num_tasks dirichlet_samples task_dim"]
    boundaries: Array  # int32, shape (num_tasks, dirichlet_samples+1)
    dirichlet_samples: int


def make_dirichlet_reset_schedule(
    key: PRNGKeyArray,
    *,
    num_tasks: int,
    task_dim: int,
    dirichlet_samples: int,
    num_steps: int,
) -> DirichletResetSchedule:
    assert dirichlet_samples >= 1

    key_p, key_tasks = jax.random.split(key, 2)

    p = _dirichlet_alpha1(key_p, (num_tasks, dirichlet_samples))
    p_np = np.asarray(jax.device_get(p))
    boundaries_np = np.zeros((num_tasks, dirichlet_samples + 1), dtype=np.int32)
    for i in range(num_tasks):
        lengths = _proportions_to_integer_lengths(p_np[i], num_steps)
        boundaries_np[i, 0] = 0
        boundaries_np[i, 1:] = np.cumsum(lengths, dtype=np.int64).astype(np.int32)
        boundaries_np[i, -1] = np.int32(num_steps)

    task_bank = jax.random.normal(
        key_tasks,
        shape=(num_tasks, dirichlet_samples, task_dim),
    )

    return DirichletResetSchedule(
        tasks=task_bank,
        boundaries=jnp.asarray(boundaries_np, dtype=jnp.int32),
        dirichlet_samples=dirichlet_samples,
    )


def main(
    # experiment
    task_diversity: int = 1,
    mala_step_size: float = 0.0,
    num_resamples: int = 0,
    seed: int = 42,
    output_root: Path = Path("runs/iclr_task_diversity"),
    run_name: str | None = None,
    # data
    task_dim: int = 8,
    num_examples: int = 16,
    noise_var: float = 0.25,
    # model
    head_type: Literal["point", "gaussian", "mog"] = "point",
    num_components: int = 4,
    num_blocks: int = 8,
    num_heads: int = 2,
    embed_size: int = 128,
    mlp_size: int = 128,
    # training
    num_steps: int = 1024 * 512,
    learning_rate: float = 0.003,
    batch_size: int = 256,
    lr_warmup: bool = True,
    # eval
    eval_period: int = 64,
    eval_batch_size: int = 1024,
    delta_prompt_length: int = 16,
    delta_tail_start: int | None = None,
    checkpoint_period: int = 8192,
    # energy distance
    compute_energy_distance: bool = False,
    energy_distance_period: int = 800,
    energy_n_samples: int = 5000,
    # misc
    final_checkpoint: bool = True,
    force_restart: bool = False,
):
    """Train an in-context regression transformer at one configuration.

    Stationary by default. Set --mala-step-size for MALA random-walk non-stationarity, or --num-resamples for Dirichlet resampling non-stationarity. With --compute-energy-distance, also records the energy distance between the transformer's implicit prior (sampled via predictive Monte Carlo) and the dMMSE / ridge priors.

    Writes per-step metrics to <output_root>/<run_name>.json and an orbax checkpoint under <output_root>/checkpoints/<run_name>/. Re-invoking with the same --run-name resumes training from the latest saved checkpoint, pass --force-restart to discard this checkpoint and start again with the same run_name.

    Args:
        task_diversity: number of latent task vectors (M); task set is sampled from N(0, I_D) at start.
        mala_step_size: MALA random-walk step size (gamma) for non-stationary training; mutually exclusive with --num-resamples.
        num_resamples: number of Dirichlet resampling events (R) per task; mutually exclusive with --mala-step-size.
        seed: PRNG seed.
        output_root: parent directory under which the run JSON and checkpoints subdirectory are created.
        run_name: name for <output_root>/<run_name>.json and <output_root>/checkpoints/<run_name>/; auto-generated from a timestamp if unset. Re-invoking with the same name resumes training from the latest saved checkpoint.
        task_dim: task vector dimension (D in the paper).
        num_examples: in-context examples per sequence (K).
        noise_var: observation noise variance (sigma^2).
        head_type: prediction head. "point" predicts a scalar; "gaussian" predicts a mean and variance; "mog" predicts a mixture of Gaussians.
        num_components: number of mixture components (G); used only when --head-type=mog.
        num_blocks: number of transformer layers.
        num_heads: attention heads per layer.
        embed_size: transformer embedding dimension.
        mlp_size: MLP hidden dimension per block.
        num_steps: total training steps.
        learning_rate: Adam peak learning rate.
        batch_size: gradient batch size.
        lr_warmup: linearly warm up the LR from 0 over the first 10% of training before holding it constant; set --no-lr-warmup for a constant LR throughout.
        eval_period: interval (in training steps) between evaluation passes.
        eval_batch_size: batch size for held-out evaluation sequences (drawn from N(0, I_D)).
        delta_prompt_length: prompt prefix length used for the Delta_PT metric computations.
        delta_tail_start: if set, additionally compute Delta metrics over positions delta_tail_start..K-1.
        checkpoint_period: interval (in training steps) between orbax checkpoint saves.
        compute_energy_distance: also compute energy distance to the dMMSE and ridge priors at each evaluation step.
        energy_distance_period: interval (in training steps) between energy-distance evaluations.
        energy_n_samples: predictive Monte Carlo samples per evaluation.
        final_checkpoint: keep the orbax checkpoint directory after training finishes; set --no-final-checkpoint to delete it and retain only the JSON.
        force_restart: remove any existing checkpoint at --run-name before starting and train from scratch; default is to resume.
    """
    args = " ".join(sys.argv)
    start_time = datetime.datetime.now()

    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    assert not (mala_step_size > 0 and num_resamples > 0), (
        "cannot enable both MALA (--mala-step-size) and Dirichlet resampling "
        "(--num-resamples) at the same time"
    )

    # energy distance requires gaussian or mog head for predictive resampling
    if compute_energy_distance:
        assert head_type in (
            "gaussian",
            "mog",
        ), "energy distance requires head_type to be 'gaussian' or 'mog'"
    if delta_tail_start is not None:
        assert 0 <= delta_tail_start < num_examples, (
            f"delta_tail_start ({delta_tail_start}) must be in [0, {num_examples - 1}]"
        )

    print("configuration:")
    config = locals()
    for config_key, config_value in config.items():
        print(f"* {config_key:30s}: {config_value!r}")
    key = jax.random.PRNGKey(seed=seed)

    print("initialising training distribution...")
    key_tasks, key = jax.random.split(key)
    if num_resamples > 0:
        dirichlet_schedule = make_dirichlet_reset_schedule(
            key_tasks,
            num_tasks=task_diversity,
            task_dim=task_dim,
            dirichlet_samples=num_resamples,
            num_steps=num_steps,
        )
        tasks = DiscreteTaskDistribution(tasks=dirichlet_schedule.tasks[:, 0, :])
    else:
        dirichlet_schedule = None
        tasks = DiscreteTaskDistribution.init(
            key=key_tasks,
            task_dim=task_dim,
            num_tasks=task_diversity,
        )
    train_data = RegressionSequenceDistribution(
        task_distribution=tasks,
        noise_var=noise_var,
    )
    print(strux.tree_size(train_data), "parameters")

    print("initialising evaluation distribution...")
    test_data = RegressionSequenceDistribution(
        task_distribution=GaussianTaskDistribution(task_dim=task_dim),
        noise_var=noise_var,
    )
    print(strux.tree_size(test_data), "parameters")

    print("generating eval batch...")
    key_eval_data, key = jax.random.split(key)
    xs, ys = test_data.get_batch(
        key=key_eval_data,
        num_examples=num_examples,
        batch_size=eval_batch_size,
    )

    print("initialising model...")
    key_model, key = jax.random.split(key)
    model = InContextRegressionTransformer.init(
        key=key_model,
        task_dim=task_dim,
        num_examples=num_examples,
        head_type=head_type,
        num_components=num_components,
        num_blocks=num_blocks,
        num_heads=num_heads,
        embed_size=embed_size,
        mlp_size=mlp_size,
    )
    print(strux.tree_size(model), "parameters")

    print("initialising optimiser and lr scheduler")
    if lr_warmup:
        warmup_steps = max(1, num_steps // 10)
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=learning_rate,
                    transition_steps=warmup_steps,
                ),
                optax.constant_schedule(learning_rate),
            ],
            boundaries=[warmup_steps],
        )
    else:
        lr_schedule = optax.schedules.constant_schedule(learning_rate)
    optimiser = optax.adam(
        learning_rate=lr_schedule,
    )
    opt_state = optimiser.init(model)

    print("defining updates...")

    def update_data(state: TrainState) -> TrainState:
        key, subkey = jax.random.split(state.key)

        if num_resamples > 0:
            boundaries = dirichlet_schedule.boundaries
            idxs = jnp.sum(state.step >= boundaries[:, 1:], axis=1).astype(jnp.int32)
            gathered = jnp.take_along_axis(
                dirichlet_schedule.tasks,
                idxs[:, None, None],
                axis=1,
            )
            tasks_now = gathered[:, 0, :]
            train_data = state.train_data
            new_dist = train_data.task_distribution.replace(tasks=tasks_now)
            train_data = train_data.replace(task_distribution=new_dist)
            return state.replace(train_data=train_data, key=key)

        if mala_step_size == 0:
            return state.replace(key=key)

        def mala(key, task, step_size):
            # langevin step proposal
            key_noise, key = jax.random.split(key)
            noise = jax.random.normal(key=key_noise, shape=task.shape)
            proposal = (1 - step_size / 2) * task + jnp.sqrt(step_size) * noise
            # metropolis--hastings acceptance test
            score = step_size / 8 * (jnp.sum(proposal**2) - jnp.sum(task**2))
            p_accept = jnp.minimum(1.0, jnp.exp(-score))
            key_accept, key = jax.random.split(key)
            accept = jax.random.bernoulli(
                key_accept,
                p=p_accept,
                shape=(),
            )
            return jnp.where(accept, proposal, task)

        train_data = state.train_data
        tasks = jax.vmap(
            mala,
            in_axes=(0, 0, None),
        )(
            jax.random.split(subkey, train_data.task_distribution.num_tasks),
            train_data.task_distribution.tasks,
            mala_step_size,
        )
        task_distribution = train_data.task_distribution.replace(tasks=tasks)
        train_data = train_data.replace(task_distribution=task_distribution)

        return state.replace(train_data=train_data, key=key)

    def update_model(state: TrainState) -> TrainState:
        key, _ = jax.random.split(state.key)

        xs_batch, ys_batch = state.train_data.get_batch(
            key,
            num_examples=num_examples,
            batch_size=batch_size,
        )

        def loss_fn(
            model: InContextRegressionTransformer,
            xs: Float[Array, "batch_size num_examples task_dim"],
            ys: Float[Array, "batch_size num_examples"],
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            if model.head_type == "gaussian":
                ys_pred, ys_var = model.forward_gaussian_batch(
                    xs, ys, noise_var=noise_var
                )
                mse = jnp.mean((ys_pred - ys) ** 2)
                nll = jnp.mean(gaussian_nll(ys, ys_pred, ys_var))
                return nll, mse
            elif model.head_type == "mog":
                logits, means, vars = model.forward_mog_batch(xs, ys)
                nll = jnp.mean(mog_nll(ys, logits, means, vars))
                # MSE uses mixture mean: sum(pi_k * mu_k)
                pi = jax.nn.softmax(logits, axis=-1)
                ys_pred = (pi * means).sum(axis=-1)
                mse = jnp.mean((ys_pred - ys) ** 2)
                return nll, mse
            else:
                ys_pred = model.forward_point_batch(xs, ys)
                mse = jnp.mean((ys_pred - ys) ** 2)
                return mse, mse

        (loss, mse), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model, xs_batch, ys_batch
        )
        updates, new_opt_state = optimiser.update(grads, state.opt_state, state.model)
        new_model = optax.apply_updates(state.model, updates)

        new_state = state.replace(
            model=new_model,
            opt_state=new_opt_state,
            key=key,
            step=state.step + 1,
        )
        return new_state, loss, mse, xs_batch, ys_batch

    @functools.partial(
        jax.jit,
        static_argnames=["num_steps"],
        donate_argnums=(0,),
    )
    def train_steps(
        state: TrainState,
        num_steps: int,
    ) -> tuple[
        TrainState,
        Float[Array, ""],
        Float[Array, ""],
        Float[Array, "batch_size num_examples task_dim"],
        Float[Array, "batch_size num_examples"],
    ]:
        init_loss = jnp.array(0.0)
        init_mse = jnp.array(0.0)
        init_xs = jnp.zeros((batch_size, num_examples, task_dim))
        init_ys = jnp.zeros((batch_size, num_examples))

        def body(
            _i: int,
            carry: tuple[
                TrainState,
                Float[Array, ""],
                Float[Array, ""],
                Float[Array, "batch_size num_examples task_dim"],
                Float[Array, "batch_size num_examples"],
            ],
        ) -> tuple[
            TrainState,
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, "batch_size num_examples task_dim"],
            Float[Array, "batch_size num_examples"],
        ]:
            state, loss, mse, xs_batch, ys_batch = carry
            state = update_data(state)
            state, loss, mse, xs_batch, ys_batch = update_model(state)
            return state, loss, mse, xs_batch, ys_batch

        return jax.lax.fori_loop(
            0,
            num_steps,
            body,
            (state, init_loss, init_mse, init_xs, init_ys),
        )

    output_root_abs = os.path.abspath(output_root)
    run_path = os.path.join(output_root_abs, f"{run_name}.json")
    checkpoint_dir = os.path.join(output_root_abs, "checkpoints", run_name)

    if force_restart:
        if os.path.exists(run_path):
            print(f"Force restart: removing existing run file {run_path}")
            os.remove(run_path)
        if os.path.exists(checkpoint_dir):
            print(f"Force restart: removing existing checkpoint dir {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)

    if os.path.exists(run_path):
        print(f"skipping {run_name}: {run_path} already exists")
        return

    state = TrainState(
        model=model,
        opt_state=opt_state,
        train_data=train_data,
        key=key,
        step=0,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    with ocp.CheckpointManager(
        checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            create=True,
            enable_async_checkpointing=True,
        ),
    ) as manager:
        latest_step = manager.latest_step()
        if latest_step is not None:
            print(f"Restoring from checkpoint at step {latest_step}...")
            restored = manager.restore(
                latest_step,
                args=ocp.args.StandardRestore(item=state),
            )
            state = TrainState(**restored) if isinstance(restored, dict) else restored
        else:
            state = TrainState(
                model=model,
                opt_state=opt_state,
                train_data=train_data,
                key=key,
                step=0,
            )

        print("starting training loop...")

        stats = []

        start_step = int(state.step)
        loss = 0.0

        if start_step >= num_steps:
            print("training run already finished")
            return

        pbar = tqdm(
            total=num_steps,
            desc="Training",
            initial=start_step,
        )

        t = start_step
        last_energy_distance_step = (
            start_step - energy_distance_period
        )  # ensure first eval runs
        while t < num_steps:
            if t % eval_period == 0:
                next_eval = t
            else:
                next_eval = t + (eval_period - (t % eval_period))
            if next_eval >= num_steps:
                next_eval = num_steps - 1

            steps_this = next_eval - t + 1
            state, loss, train_mse, xs_batch, ys_batch = train_steps(state, steps_this)
            pbar.update(steps_this)
            t = next_eval

            current_tasks = state.train_data.task_distribution.tasks
            lr_t = lr_schedule(t)
            # compute all eval metrics in single JIT call
            if head_type == "point":
                eval_metrics = _compute_eval_metrics_point_jit(
                    state.model,
                    xs,
                    ys,
                    xs_batch,
                    ys_batch,
                    current_tasks,
                    current_tasks,
                    noise_var,
                    task_dim=task_dim,
                    delta_prompt_length=delta_prompt_length,
                    delta_tail_start=delta_tail_start,
                )
            elif head_type == "gaussian":
                eval_metrics = _compute_eval_metrics_gaussian_jit(
                    state.model,
                    xs,
                    ys,
                    xs_batch,
                    ys_batch,
                    current_tasks,
                    current_tasks,
                    noise_var,
                    task_dim=task_dim,
                    delta_prompt_length=delta_prompt_length,
                    delta_tail_start=delta_tail_start,
                )
            else:  # mog
                eval_metrics = _compute_eval_metrics_mog_jit(
                    state.model,
                    xs,
                    ys,
                    xs_batch,
                    ys_batch,
                    current_tasks,
                    current_tasks,
                    noise_var,
                    task_dim=task_dim,
                    delta_prompt_length=delta_prompt_length,
                    delta_tail_start=delta_tail_start,
                )

            step_stats: dict = {
                "step": t,
                "lr": lr_t,
                "train_error_model": train_mse,
                "train_error_ridge": eval_metrics["train_error_ridge"],
                "train_error_dmmse": eval_metrics["train_error_dmmse"],
                "test_error_ridge": eval_metrics["test_error_ridge"],
                "test_error_dmmse": eval_metrics["test_error_dmmse"],
                "test_error_model": eval_metrics["test_error"],
                "train_delta_ridge": eval_metrics["train_delta_ridge"],
                "train_delta_dmmse": eval_metrics["train_delta_dmmse"],
                "test_delta_ridge": eval_metrics["test_delta_ridge"],
                "test_delta_dmmse": eval_metrics["test_delta_dmmse"],
                "timestamp": datetime.datetime.now().timestamp(),
            }
            if delta_tail_start is not None:
                step_stats["train_delta_tail_ridge"] = eval_metrics[
                    "train_delta_tail_ridge"
                ]
                step_stats["train_delta_tail_dmmse"] = eval_metrics[
                    "train_delta_tail_dmmse"
                ]
                step_stats["test_delta_tail_ridge"] = eval_metrics[
                    "test_delta_tail_ridge"
                ]
                step_stats["test_delta_tail_dmmse"] = eval_metrics[
                    "test_delta_tail_dmmse"
                ]

            # add NLL for gaussian and mog heads
            if head_type in ("gaussian", "mog"):
                step_stats["train_nll_model"] = loss
                step_stats["test_nll_model"] = eval_metrics["test_nll"]

            # compute energy distance against the dMMSE / ridge priors
            if (
                compute_energy_distance
                and (t - last_energy_distance_step) >= energy_distance_period
            ):
                last_energy_distance_step = t

                key_ed, new_key = jax.random.split(state.key)
                state = state.replace(key=new_key)
                current_tasks = state.train_data.task_distribution.tasks

                key_pr, key_ridge, key_dmmse = jax.random.split(key_ed, 3)
                prior_task_estimates = montecarlo_task_estimate(
                    model=state.model,
                    key=key_pr,
                    num_samples=energy_n_samples,
                    num_steps=num_examples,
                    noise_var=noise_var,
                    print_progress=False,
                )

                ridge_samples = jax.random.normal(
                    key_ridge, (energy_n_samples, task_dim)
                )
                indices = jax.random.randint(
                    key_dmmse, (energy_n_samples,), 0, len(current_tasks)
                )
                dmmse_samples = current_tasks[indices]

                step_stats["energy_dist/prior/ridge"] = energy_distance_multidim(
                    prior_task_estimates, ridge_samples
                )
                step_stats["energy_dist/prior/dmmse"] = energy_distance_multidim(
                    prior_task_estimates, dmmse_samples
                )

            # materialize metrics with single sync point for logging
            step_stats_materialized = materialize_metrics(step_stats)
            stats.append(step_stats_materialized)

            steps_per_duration = (datetime.datetime.now() - start_time) / (
                t - start_step + 1
            )
            steps_remaining = num_steps - t
            est_time_remaining = steps_remaining * steps_per_duration

            if head_type in ("gaussian", "mog"):
                print(
                    f"step {t}: lr={lr_t:.4f} "
                    f"loss_nll={step_stats_materialized['train_nll_model']:.4f} "
                    f"train_mse={step_stats_materialized['train_error_model']:.4f} "
                    f"test_mse={step_stats_materialized['test_error_model']:.4f} "
                    f"test_nll={step_stats_materialized['test_nll_model']:.4f} "
                    f"est_time_remaining={est_time_remaining}"
                )
            else:
                print(
                    f"step {t}: lr={lr_t:.4f} "
                    f"loss_mse={step_stats_materialized['train_error_model']:.4f} "
                    f"train_mse={step_stats_materialized['train_error_model']:.4f} "
                    f"test_mse={step_stats_materialized['test_error_model']:.4f} "
                    f"est_time_remaining={est_time_remaining}"
                )

            # orbax checkpoint: save whole TrainState
            if t % checkpoint_period == 0 or t == num_steps - 1:
                manager.save(
                    t,
                    args=ocp.args.StandardSave(state),
                )

            t += 1

        pbar.close()
        print("done!")
        manager.wait_until_finished()

        end_time = datetime.datetime.now()

        print(f"saving results to {run_path!r}...")
        data: dict = {
            "times": {
                "start": start_time.timestamp(),
                "end": end_time.timestamp(),
            },
            "config": config,
            "stats": stats,
        }

        os.makedirs(os.path.dirname(run_path), exist_ok=True)
        with open(run_path, "w") as outfile:
            json.dump(obj=data, fp=outfile, indent=2, default=str)

        if not final_checkpoint and os.path.exists(checkpoint_dir):
            print(f"removing checkpoint dir {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)


if __name__ == "__main__":
    tyro.cli(main)
