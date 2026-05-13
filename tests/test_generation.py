"""Tests for generation.py - run with: uv run python -m tests.test_generation

Small model + short runs, fast on CPU; a TPU is not required."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from iclr import InContextRegressionTransformer, generation


def make_model(head_type: str = "gaussian", task_dim: int = 4, seed: int = 0):
    return InContextRegressionTransformer.init(
        key=jax.random.PRNGKey(seed),
        task_dim=task_dim,
        num_examples=8,
        num_blocks=2,
        num_heads=2,
        embed_size=32,
        mlp_size=32,
        head_type=head_type,
        num_components=4,
    )


# 1. Shape correctness


def test_single_unconditional_shapes():
    model = make_model()
    xs, ys = generation.generate_single(
        model,
        jax.random.PRNGKey(0),
        num_examples=8,
        noise_var=0.125,
    )
    assert xs.shape == (8, 4), f"xs shape: {xs.shape}"
    assert ys.shape == (8,), f"ys shape: {ys.shape}"
    assert jnp.isfinite(xs).all()
    assert jnp.isfinite(ys).all()
    print("test_single_unconditional_shapes PASSED")


def test_single_conditioned_shapes():
    model = make_model()
    prompt_xs = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    prompt_ys = jnp.array([1.0, 2.0, 3.0])

    xs, ys = generation.generate_single(
        model,
        jax.random.PRNGKey(2),
        num_examples=8,
        noise_var=0.125,
        prompt_xs=prompt_xs,
        prompt_ys=prompt_ys,
    )
    assert xs.shape == (8, 4), f"xs shape: {xs.shape}"
    assert ys.shape == (8,), f"ys shape: {ys.shape}"
    assert jnp.isfinite(xs).all()
    assert jnp.isfinite(ys).all()
    print("test_single_conditioned_shapes PASSED")


def test_batch_unconditional_shapes():
    model = make_model()
    xs, ys = generation.generate_batch(
        model,
        jax.random.PRNGKey(0),
        num_generations=16,
        num_examples=8,
        noise_var=0.125,
    )
    assert xs.shape == (16, 8, 4), f"xs shape: {xs.shape}"
    assert ys.shape == (16, 8), f"ys shape: {ys.shape}"
    assert jnp.isfinite(xs).all()
    assert jnp.isfinite(ys).all()
    print("test_batch_unconditional_shapes PASSED")


def test_batch_conditioned_shapes():
    model = make_model()
    prompt_xs = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    prompt_ys = jnp.array([1.0, 2.0, 3.0])

    xs, ys = generation.generate_batch(
        model,
        jax.random.PRNGKey(2),
        num_generations=16,
        num_examples=8,
        noise_var=0.125,
        prompt_xs=prompt_xs,
        prompt_ys=prompt_ys,
    )
    assert xs.shape == (16, 8, 4), f"xs shape: {xs.shape}"
    assert ys.shape == (16, 8), f"ys shape: {ys.shape}"
    assert jnp.isfinite(xs).all()
    assert jnp.isfinite(ys).all()
    print("test_batch_conditioned_shapes PASSED")


def test_multi_prompt_shapes():
    model = make_model()
    prompt_xs = jax.random.normal(jax.random.PRNGKey(1), (5, 3, 4))
    prompt_ys = jax.random.normal(jax.random.PRNGKey(2), (5, 3))

    xs, ys = generation.generate_multi_prompt(
        model,
        jax.random.PRNGKey(3),
        prompt_xs=prompt_xs,
        prompt_ys=prompt_ys,
        num_generations_per_prompt=8,
        num_examples=8,
        noise_var=0.125,
    )
    assert xs.shape == (5, 8, 8, 4), f"xs shape: {xs.shape}"
    assert ys.shape == (5, 8, 8), f"ys shape: {ys.shape}"
    assert jnp.isfinite(xs).all()
    assert jnp.isfinite(ys).all()
    print("test_multi_prompt_shapes PASSED")


# 2. Prompt preservation


def test_single_prompt_preserved():
    model = make_model()
    prompt_xs = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    prompt_ys = jnp.array([1.0, 2.0, 3.0])

    xs, ys = generation.generate_single(
        model,
        jax.random.PRNGKey(2),
        num_examples=8,
        noise_var=0.125,
        prompt_xs=prompt_xs,
        prompt_ys=prompt_ys,
    )

    assert jnp.allclose(xs[:3, :], prompt_xs), "prompt_xs not preserved"
    assert jnp.allclose(ys[:3], prompt_ys), "prompt_ys not preserved"
    print("test_single_prompt_preserved PASSED")


def test_batch_prompt_preserved():
    model = make_model()
    prompt_xs = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    prompt_ys = jnp.array([1.0, 2.0, 3.0])

    xs, ys = generation.generate_batch(
        model,
        jax.random.PRNGKey(2),
        num_generations=16,
        num_examples=8,
        noise_var=0.125,
        prompt_xs=prompt_xs,
        prompt_ys=prompt_ys,
    )

    for i in range(16):
        assert jnp.allclose(xs[i, :3, :], prompt_xs), (
            f"prompt_xs not preserved in gen {i}"
        )
        assert jnp.allclose(ys[i, :3], prompt_ys), f"prompt_ys not preserved in gen {i}"
    print("test_batch_prompt_preserved PASSED")


# 3. Determinism


def test_single_determinism():
    model = make_model()
    key = jax.random.PRNGKey(42)

    xs1, ys1 = generation.generate_single(model, key, num_examples=8, noise_var=0.125)
    xs2, ys2 = generation.generate_single(model, key, num_examples=8, noise_var=0.125)

    assert jnp.allclose(xs1, xs2), "xs not deterministic"
    assert jnp.allclose(ys1, ys2), "ys not deterministic"
    print("test_single_determinism PASSED")


def test_batch_determinism():
    model = make_model()
    key = jax.random.PRNGKey(42)

    xs1, ys1 = generation.generate_batch(
        model, key, num_generations=8, num_examples=8, noise_var=0.125
    )
    xs2, ys2 = generation.generate_batch(
        model, key, num_generations=8, num_examples=8, noise_var=0.125
    )

    assert jnp.allclose(xs1, xs2), "xs not deterministic"
    assert jnp.allclose(ys1, ys2), "ys not deterministic"
    print("test_batch_determinism PASSED")


# 4. Validation


def test_task_size_mismatch():
    model = make_model(task_dim=4)
    prompt_xs = jnp.ones((2, 3))  # wrong: 3 instead of 4
    prompt_ys = jnp.ones((2,))

    try:
        generation.generate_batch(
            model,
            jax.random.PRNGKey(0),
            num_generations=4,
            num_examples=8,
            noise_var=0.125,
            prompt_xs=prompt_xs,
            prompt_ys=prompt_ys,
        )
        raise RuntimeError("Should have raised AssertionError")
    except AssertionError:
        pass
    print("test_task_size_mismatch PASSED")


# 5. All head types


def test_point_head():
    model = make_model(head_type="point")
    xs, ys = generation.generate_batch(
        model,
        jax.random.PRNGKey(0),
        num_generations=8,
        num_examples=8,
        noise_var=0.125,
    )
    assert xs.shape == (8, 8, 4)
    assert ys.shape == (8, 8)
    assert jnp.isfinite(xs).all()
    assert jnp.isfinite(ys).all()
    print("test_point_head PASSED")


def test_gaussian_head():
    model = make_model(head_type="gaussian")
    xs, ys = generation.generate_batch(
        model,
        jax.random.PRNGKey(0),
        num_generations=8,
        num_examples=8,
        noise_var=0.125,
    )
    assert xs.shape == (8, 8, 4)
    assert ys.shape == (8, 8)
    assert jnp.isfinite(xs).all()
    assert jnp.isfinite(ys).all()
    print("test_gaussian_head PASSED")


def test_mog_head():
    model = make_model(head_type="mog")
    xs, ys = generation.generate_batch(
        model,
        jax.random.PRNGKey(0),
        num_generations=8,
        num_examples=8,
        noise_var=0.125,
    )
    assert xs.shape == (8, 8, 4)
    assert ys.shape == (8, 8)
    assert jnp.isfinite(xs).all()
    assert jnp.isfinite(ys).all()
    print("test_mog_head PASSED")


# 6. Vmap consistency


def test_batch_vs_loop():
    """Verify generate_batch == loop over generate_single."""
    model = make_model()
    num_generations = 4
    num_examples = 8

    key = jax.random.PRNGKey(0)

    xs_batch, ys_batch = generation.generate_batch(
        model,
        key,
        num_generations=num_generations,
        num_examples=num_examples,
        noise_var=0.125,
    )

    keys = jax.random.split(key, num_generations)
    xs_loop = []
    ys_loop = []
    for i in range(num_generations):
        xs_i, ys_i = generation.generate_single(
            model,
            keys[i],
            num_examples=num_examples,
            noise_var=0.125,
        )
        xs_loop.append(xs_i)
        ys_loop.append(ys_i)
    xs_loop = jnp.stack(xs_loop)
    ys_loop = jnp.stack(ys_loop)

    assert jnp.allclose(xs_batch, xs_loop), "batch vs loop xs mismatch"
    assert jnp.allclose(ys_batch, ys_loop), "batch vs loop ys mismatch"
    print("test_batch_vs_loop PASSED")


def test_multi_prompt_vs_loop():
    """Verify generate_multi_prompt == loop over generate_batch."""
    model = make_model()
    num_prompts = 3
    num_generations = 4
    num_examples = 8
    task_dim = 4
    prompt_len = 2

    key = jax.random.PRNGKey(0)
    key_prompts, key_multi = jax.random.split(key)

    k1, k2 = jax.random.split(key_prompts)
    prompt_xs = jax.random.normal(k1, (num_prompts, prompt_len, task_dim))
    prompt_ys = jax.random.normal(k2, (num_prompts, prompt_len))

    xs_multi, ys_multi = generation.generate_multi_prompt(
        model,
        key_multi,
        prompt_xs=prompt_xs,
        prompt_ys=prompt_ys,
        num_generations_per_prompt=num_generations,
        num_examples=num_examples,
        noise_var=0.125,
    )

    keys = jax.random.split(key_multi, num_prompts)
    xs_loop = []
    ys_loop = []
    for i in range(num_prompts):
        xs_i, ys_i = generation.generate_batch(
            model,
            keys[i],
            num_generations=num_generations,
            num_examples=num_examples,
            noise_var=0.125,
            prompt_xs=prompt_xs[i],
            prompt_ys=prompt_ys[i],
        )
        xs_loop.append(xs_i)
        ys_loop.append(ys_i)
    xs_loop = jnp.stack(xs_loop)
    ys_loop = jnp.stack(ys_loop)

    assert jnp.allclose(xs_multi, xs_loop), "multi-prompt xs mismatch"
    assert jnp.allclose(ys_multi, ys_loop), "multi-prompt ys mismatch"
    print("test_multi_prompt_vs_loop PASSED")


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    print()

    test_single_unconditional_shapes()
    test_single_conditioned_shapes()
    test_batch_unconditional_shapes()
    test_batch_conditioned_shapes()
    test_multi_prompt_shapes()
    test_single_prompt_preserved()
    test_batch_prompt_preserved()
    test_single_determinism()
    test_batch_determinism()
    test_task_size_mismatch()
    test_point_head()
    test_gaussian_head()
    test_mog_head()
    test_batch_vs_loop()
    test_multi_prompt_vs_loop()

    print()
    print("All tests PASSED")
