"""Train a transformer and snapshot its implicit prior over task vectors via predictive Monte Carlo.

Defaults configure the 1D scenario (task_dim=1, task_diversity=1) where the prior can be visualised as a histogram on the real line. Stationary by default. Set --mala-step-size for MALA random-walk non-stationarity, or --num-resamples for equispaced resampling non-stationarity (this is the equispaced setting; the Dirichlet partition lives in experiment_iclr_task_diversity.py).

Snapshots are taken every --snapshot-period training steps; an optional dense-snapshot window (--dense-step-start, --dense-step-end, --dense-snapshot-period) takes higher-resolution snapshots within a sub-interval, useful for visualising fine dynamics around a particular stretch of training.

Writes a snapshots.npz archive of prior samples and per-snapshot Delta_PT,dMMSE / Delta_PT,Ridge metrics under <output_root>/<timestamp>/. Each invocation creates a new timestamped subdirectory; this script does not resume from previous runs.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm

import iclr
from iclr import dmmse_batch, montecarlo_task_estimate, ridge_batch


@dataclasses.dataclass(frozen=True)
class Config:
    """Train a transformer and snapshot its implicit prior over task vectors via predictive Monte Carlo.

    Defaults configure the 1D scenario (task_dim=1, task_diversity=1) where the prior can be visualised as a histogram on the real line. Stationary by default. Set --mala-step-size for MALA random-walk non-stationarity, or --num-resamples for equispaced resampling non-stationarity (this is the equispaced scheme; the Dirichlet partition lives in experiment_iclr_task_diversity.py). MALA takes precedence when both are set.

    Snapshots are taken every --snapshot-period training steps; an optional dense-snapshot window (--dense-step-start, --dense-step-end, --dense-snapshot-period) takes higher-resolution snapshots within a sub-interval.

    Writes a snapshots.npz archive of prior samples and per-snapshot Delta_PT,dMMSE / Delta_PT,Ridge metrics under <output_root>/<run_name>/, where <run_name> defaults to a timestamp if not specified. This script does not resume from previous runs.
    """

    # ----- experiment -----

    # number of latent task vectors (M); fixed set sampled from N(0, I_D)
    task_diversity: int = 1
    # MALA random-walk step size (gamma) for non-stationary training; takes precedence over --num-resamples if both are set
    mala_step_size: float = 0.0
    # number of independently-sampled task sets shown over R equal-length
    # segments (R-1 internal resampling events); 0 disables resampling
    num_resamples: int = 0
    # PRNG seed
    seed: int = 0
    # parent directory under which the run subdirectory is created
    output_root: Path = Path("runs/iclr_prior_tracking")
    # name for the run subdirectory under output_root; defaults to a timestamp if unset
    run_name: str | None = None

    # ----- data -----

    # task vector dimension (D)
    task_dim: int = 1
    # in-context examples per sequence (K)
    num_examples: int = 64
    # observation noise variance (sigma^2)
    noise_var: float = 0.25

    # ----- model -----

    # prediction head: single Gaussian (mean+variance) or mixture of Gaussians
    head_type: Literal["gaussian", "mog"] = "gaussian"
    # number of mixture components (G); used only when --head-type=mog
    num_components: int = 4
    # number of transformer layers
    num_blocks: int = 8
    # attention heads per layer
    num_heads: int = 2
    # transformer embedding dimension
    embed_size: int = 128
    # MLP hidden dimension per block
    mlp_size: int = 128

    # ----- training -----

    # total training steps
    num_steps: int = 524_288
    # Adam peak learning rate
    learning_rate: float = 3e-3
    # gradient batch size
    batch_size: int = 256

    # ----- snapshots -----

    # interval (in training steps) between uniform snapshots
    snapshot_period: int = 512
    # predictive Monte Carlo samples per snapshot
    num_pr_samples: int = 200
    # autoregressive rollout length per PMC sample
    pr_generation_steps: int = 16
    # interval between dense snapshots within the dense window; if unset, only uniform snapshots are taken
    dense_snapshot_period: int | None = None
    # start of an explicit dense-snapshot window; overrides the resampling-aligned dense window
    dense_step_start: int | None = None
    # end of an explicit dense-snapshot window
    dense_step_end: int | None = None
    # width of the dense window placed around each resampling event (used when dense_step_start/end are unset)
    dense_snapshot_window: int = 2048

    # ----- misc -----

    # remove any existing snapshots.npz at --run-name before starting; default is to refuse to overwrite
    force_restart: bool = False


def snapshot_delta_metrics(
    model: iclr.InContextRegressionTransformer,
    tasks: Float[Array, "M task_dim"],
    key: PRNGKeyArray,
    *,
    batch_size: int,
    num_examples: int,
    task_dim: int,
    noise_var: float,
) -> tuple[float, float]:
    """Sample an in-distribution evaluation batch under q_M^{(τ)}, then return
    (Δ_PT,dMMSE, Δ_PT,ridge): the mean squared difference between the model's
    predictions and the dMMSE / ridge reference predictions, normalised by D."""
    M = tasks.shape[0]
    key_x, key_idx, key_eps = jax.random.split(key, 3)
    xs = jax.random.normal(key_x, (batch_size, num_examples, task_dim))
    idx = jax.random.randint(key_idx, (batch_size,), 0, M)
    selected = tasks[idx]
    eps = jax.random.normal(key_eps, (batch_size, num_examples)) * jnp.sqrt(noise_var)
    ys = jnp.einsum("bd,bkd->bk", selected, xs) + eps

    ys_pred, _ = model.forward_gaussian_batch(xs, ys, noise_var=noise_var)
    ys_pred_dmmse = dmmse_batch(xs, ys, tasks, noise_var)
    ys_pred_ridge = ridge_batch(xs, ys, noise_var)

    delta_dmmse = ((ys_pred - ys_pred_dmmse) ** 2).mean() / task_dim
    delta_ridge = ((ys_pred - ys_pred_ridge) ** 2).mean() / task_dim
    return float(delta_dmmse), float(delta_ridge)


def snapshot_pr_prior(
    model: iclr.InContextRegressionTransformer,
    key: PRNGKeyArray,
    num_samples: int,
    num_steps: int,
    noise_var: float,
) -> Float[Array, "num_samples task_dim"]:
    return montecarlo_task_estimate(
        model,
        key,
        num_samples=num_samples,
        num_steps=num_steps,
        noise_var=noise_var,
    )


def main(cfg: Config) -> None:
    run_name = cfg.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / run_name
    data_path = run_dir / "snapshots.npz"
    if data_path.exists():
        if cfg.force_restart:
            print(f"Force restart: removing existing snapshots {data_path}")
            data_path.unlink()
        else:
            raise FileExistsError(
                f"{data_path} already exists; pass --force-restart to overwrite "
                f"or choose a different --run-name (this script does not resume)."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    print(f"Run directory: {run_dir}")

    use_mala = cfg.mala_step_size > 0
    use_resampling = not use_mala and cfg.num_resamples > 0

    key = jax.random.PRNGKey(cfg.seed)

    # initialise tasks and training distribution
    key_tasks, key = jax.random.split(key)
    task_distribution = iclr.DiscreteTaskDistribution.init(
        key=key_tasks,
        task_dim=cfg.task_dim,
        num_tasks=cfg.task_diversity,
    )
    train_data = iclr.RegressionSequenceDistribution(
        task_distribution=task_distribution,
        noise_var=cfg.noise_var,
    )

    # initialise model
    key_model, key = jax.random.split(key)
    model = iclr.InContextRegressionTransformer.init(
        key=key_model,
        task_dim=cfg.task_dim,
        num_examples=cfg.num_examples,
        num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads,
        embed_size=cfg.embed_size,
        mlp_size=cfg.mlp_size,
        head_type=cfg.head_type,
        num_components=cfg.num_components,
    )

    # optimiser with 10% warmup, no decay (constant after warmup)
    warmup_steps = max(1, cfg.num_steps // 10)
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=cfg.learning_rate,
                transition_steps=warmup_steps,
            ),
            optax.constant_schedule(cfg.learning_rate),
        ],
        boundaries=[warmup_steps],
    )
    optimiser = optax.adam(learning_rate=lr_schedule)
    opt_state = optimiser.init(model)

    state = iclr.TrainState(
        model=model,
        opt_state=opt_state,
        train_data=train_data,
        key=key,
        step=0,
    )

    # MALA update function
    if use_mala:
        mala_step_size = cfg.mala_step_size

        @jax.jit
        def update_data(state: iclr.TrainState) -> iclr.TrainState:
            key, subkey = jax.random.split(state.key)

            def mala(key, task, step_size):
                key_noise, key_accept = jax.random.split(key)
                noise = jax.random.normal(key=key_noise, shape=task.shape)
                proposal = (1 - step_size / 2) * task + jnp.sqrt(step_size) * noise
                score = step_size / 8 * (jnp.sum(proposal**2) - jnp.sum(task**2))
                p_accept = jnp.minimum(1.0, jnp.exp(-score))
                accept = jax.random.bernoulli(key_accept, p=p_accept, shape=())
                return jnp.where(accept, proposal, task)

            train_data = state.train_data
            tasks = jax.vmap(mala, in_axes=(0, 0, None))(
                jax.random.split(subkey, train_data.task_distribution.num_tasks),
                train_data.task_distribution.tasks,
                mala_step_size,
            )
            task_distribution = train_data.task_distribution.replace(tasks=tasks)
            train_data = train_data.replace(task_distribution=task_distribution)
            return state.replace(train_data=train_data, key=key)

    # model update function
    @jax.jit
    def update_model(
        state: iclr.TrainState,
    ) -> tuple[iclr.TrainState, Float[Array, ""]]:
        key, _ = jax.random.split(state.key)
        xs_batch, ys_batch = state.train_data.get_batch(
            key,
            num_examples=cfg.num_examples,
            batch_size=cfg.batch_size,
        )

        def loss_fn(
            model: iclr.InContextRegressionTransformer,
            xs: Float[Array, "batch_size num_examples task_dim"],
            ys: Float[Array, "batch_size num_examples"],
        ) -> Float[Array, ""]:
            if cfg.head_type == "mog":
                logits, means, vars_ = model.forward_mog_batch(xs, ys)
                return jnp.mean(iclr.mog_nll(ys, logits, means, vars_))
            ys_pred, ys_var = model.forward_gaussian_batch(
                xs, ys, noise_var=cfg.noise_var
            )
            return jnp.mean(iclr.gaussian_nll(ys, ys_pred, ys_var))

        loss, grads = jax.value_and_grad(loss_fn)(state.model, xs_batch, ys_batch)
        updates, new_opt_state = optimiser.update(grads, state.opt_state, state.model)
        new_model = optax.apply_updates(state.model, updates)
        return state.replace(
            model=new_model,
            opt_state=new_opt_state,
            key=key,
            step=state.step + 1,
        ), loss

    # compute resampling schedule: --num-resamples = R total task sets shown
    # across the run, partitioned into R equal-length segments by R-1 internal
    # transitions; the right segment-end at step num_steps is intentionally
    # excluded (it is never reached by the t-loop anyway).
    if use_resampling:
        resample_period = cfg.num_steps // cfg.num_resamples
        resample_steps = {resample_period * i for i in range(1, cfg.num_resamples)}
        print(f"Task resampling at steps: {sorted(resample_steps)}")
    else:
        resample_steps = set()
        if use_mala:
            print(f"MALA drift with step size {cfg.mala_step_size}")
        else:
            print("Stationary training (no resampling, no MALA)")

    # build snapshot schedule
    snapshot_at: set[int] = set()
    # uniform snapshots
    for t in range(0, cfg.num_steps, cfg.snapshot_period):
        snapshot_at.add(t)
    snapshot_at.add(cfg.num_steps - 1)
    # dense snapshots around resampling events — skipped when an explicit
    # [dense_step_start, dense_step_end) window is given (the explicit window
    # supersedes; otherwise resample-every-step regimes blow up to a snapshot
    # per training step, which is prohibitively slow).
    if (
        cfg.dense_snapshot_period is not None
        and use_resampling
        and cfg.dense_step_start is None
        and cfg.dense_step_end is None
    ):
        for rs in resample_steps:
            for t in range(
                rs,
                min(rs + cfg.dense_snapshot_window, cfg.num_steps),
                cfg.dense_snapshot_period,
            ):
                snapshot_at.add(t)
    # dense snapshots in an arbitrary step window
    if (
        cfg.dense_snapshot_period is not None
        and cfg.dense_step_start is not None
        and cfg.dense_step_end is not None
    ):
        for t in range(
            cfg.dense_step_start,
            min(cfg.dense_step_end, cfg.num_steps),
            cfg.dense_snapshot_period,
        ):
            snapshot_at.add(t)
    print(f"Total snapshots planned: {len(snapshot_at)}")

    # training loop
    snapshot_steps_list: list[int] = []
    snapshot_losses: list[float] = []
    snapshot_tasks: list[np.ndarray] = []
    snapshot_pr: list[np.ndarray] = []
    snapshot_delta_dmmse: list[float] = []
    snapshot_delta_ridge: list[float] = []

    key_snap = jax.random.PRNGKey(cfg.seed + 1)
    key_resample = jax.random.PRNGKey(cfg.seed + 2)
    key_delta = jax.random.PRNGKey(cfg.seed + 3)

    pbar = tqdm(range(cfg.num_steps), desc="Training")
    for t in pbar:
        # resample tasks (discrete jumps)
        if t in resample_steps:
            key_resample, subkey = jax.random.split(key_resample)
            new_tasks = jax.random.normal(
                subkey, shape=(cfg.task_diversity, cfg.task_dim)
            )
            new_dist = state.train_data.task_distribution.replace(tasks=new_tasks)
            new_train_data = state.train_data.replace(task_distribution=new_dist)
            state = state.replace(train_data=new_train_data)

        # MALA drift
        if use_mala:
            state = update_data(state)

        # gradient step
        state, loss = update_model(state)

        # snapshot
        if t in snapshot_at:
            loss_val = float(loss)
            pbar.set_postfix(loss=f"{loss_val:.4f}", snaps=len(snapshot_steps_list) + 1)

            key_snap, subkey = jax.random.split(key_snap)
            tasks_np = np.asarray(state.train_data.task_distribution.tasks)
            pr_np = np.asarray(
                snapshot_pr_prior(
                    state.model,
                    subkey,
                    num_samples=cfg.num_pr_samples,
                    num_steps=cfg.pr_generation_steps,
                    noise_var=cfg.noise_var,
                )
            )

            key_delta, subkey_d = jax.random.split(key_delta)
            d_dmmse, d_ridge = snapshot_delta_metrics(
                state.model,
                state.train_data.task_distribution.tasks,
                subkey_d,
                batch_size=256,
                num_examples=cfg.num_examples,
                task_dim=cfg.task_dim,
                noise_var=cfg.noise_var,
            )

            snapshot_steps_list.append(t)
            snapshot_losses.append(loss_val)
            snapshot_tasks.append(tasks_np)
            snapshot_pr.append(pr_np)
            snapshot_delta_dmmse.append(d_dmmse)
            snapshot_delta_ridge.append(d_ridge)

    # save
    data_path = run_dir / "snapshots.npz"
    np.savez(
        data_path,
        steps=np.asarray(snapshot_steps_list, dtype=np.int32),
        losses=np.asarray(snapshot_losses, dtype=np.float32),
        tasks=np.stack(snapshot_tasks, axis=0),
        pr=np.stack(snapshot_pr, axis=0),
        delta_dmmse=np.asarray(snapshot_delta_dmmse, dtype=np.float32),
        delta_ridge=np.asarray(snapshot_delta_ridge, dtype=np.float32),
        resample_steps=np.asarray(sorted(resample_steps), dtype=np.int32),
        config=json.dumps(dataclasses.asdict(cfg), default=str),
    )
    print(f"Saved {len(snapshot_steps_list)} snapshots to {data_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
