from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
from jaxtyping import Array, Float, PRNGKeyArray

if TYPE_CHECKING:
    from .setting import InContextRegressionTransformer


XGenerator = Callable[
    [PRNGKeyArray, tuple[int, int]],
    Float[Array, "num_examples task_dim"],
]


@functools.partial(
    jax.jit, static_argnames=["num_examples", "x_generator", "print_progress"]
)
def generate_single(
    model: InContextRegressionTransformer,
    key: PRNGKeyArray,
    *,
    num_examples: int,
    noise_var: float,
    prompt_xs: Float[Array, "prompt_len task_dim"] | None = None,
    prompt_ys: Float[Array, "prompt_len"] | None = None,
    x_generator: XGenerator = jax.random.normal,
    print_progress: bool = False,
) -> tuple[Float[Array, "num_examples task_dim"], Float[Array, "num_examples"]]:
    """
    Generate a single (xs, ys) sequence.

    Args:
        model: The transformer model
        key: PRNG key for sampling
        num_examples: Length of the sequence
        noise_var: Noise variance for gaussian/point heads
        prompt_xs: Optional prompt inputs [prompt_len, task_dim]
        prompt_ys: Optional prompt outputs [prompt_len]
        x_generator: Function to generate random x values (default: jax.random.normal)

    Returns:
        (xs, ys): Generated input-output pair
            xs: [num_examples, task_dim]
            ys: [num_examples]
    """
    task_dim = model.transformer.token_embedding.weights.shape[0] - 1
    max_capacity = model.transformer.postn_embedding.weights.shape[0] // 2
    assert num_examples <= max_capacity

    key_xs, key_ys = jax.random.split(key)
    xs = x_generator(key_xs, (num_examples, task_dim))
    ys = jnp.zeros(num_examples, dtype=xs.dtype)

    if prompt_xs is None:
        start_index = 0
    else:
        prompt_len, prompt_task_size = prompt_xs.shape
        assert prompt_task_size == task_dim
        assert prompt_ys is not None
        assert prompt_ys.shape == (prompt_len,)
        assert prompt_len <= num_examples
        start_index = prompt_len

        xs = xs.at[:prompt_len, :].set(prompt_xs)
        ys = ys.at[:prompt_len].set(prompt_ys)

    def step(
        carry: tuple[Float[Array, "num_examples"], PRNGKeyArray], t: int
    ) -> tuple[tuple[Float[Array, "num_examples"], PRNGKeyArray], None]:
        ys_curr, key_curr = carry

        if model.head_type == "mog":
            logits, means, vars_ = model.forward_mog(xs, ys_curr)
            key_curr, k_comp, k_noise = jax.random.split(key_curr, 3)
            comp = jax.random.categorical(k_comp, logits[t])
            mean_t, var_t = means[t, comp], vars_[t, comp]
        else:
            means, vars_ = model.forward_gaussian(xs, ys_curr, noise_var)
            key_curr, k_noise = jax.random.split(key_curr)
            mean_t, var_t = means[t], vars_[t]

        y_t = mean_t + jnp.sqrt(var_t) * jax.random.normal(k_noise, dtype=xs.dtype)
        return (ys_curr.at[t].set(y_t), key_curr), None

    if print_progress:
        step = scan_tqdm(num_examples - start_index)(step)

    (ys_final, _), _ = jax.lax.scan(
        step, (ys, key_ys), jnp.arange(start_index, num_examples)
    )
    return xs, ys_final


@functools.partial(
    jax.jit,
    static_argnames=[
        "num_generations",
        "num_examples",
        "x_generator",
        "print_progress",
    ],
)
def generate_batch(
    model: InContextRegressionTransformer,
    key: PRNGKeyArray,
    *,
    num_generations: int,
    num_examples: int,
    noise_var: float,
    prompt_xs: Float[Array, "prompt_len task_dim"] | None = None,
    prompt_ys: Float[Array, "prompt_len"] | None = None,
    x_generator: XGenerator = jax.random.normal,
    print_progress: bool = False,
) -> tuple[
    Float[Array, "num_generations num_examples task_dim"],
    Float[Array, "num_generations num_examples"],
]:
    """
    Generate N sequences from one prompt. Vmaps generate_single.

    Args:
        model: The transformer model
        key: PRNG key for sampling
        num_generations: Number of sequences to generate
        num_examples: Length of each sequence
        noise_var: Noise variance for gaussian/point heads
        prompt_xs: Optional prompt inputs [prompt_len, task_dim]
        prompt_ys: Optional prompt outputs [prompt_len]
        x_generator: Function to generate random x values (default: jax.random.normal)

    Returns:
        (xs, ys): Generated input-output pairs
            xs: [num_generations, num_examples, task_dim]
            ys: [num_generations, num_examples]
    """
    keys = jax.random.split(key, num_generations)
    return jax.vmap(
        lambda k: generate_single(
            model,
            k,
            num_examples=num_examples,
            noise_var=noise_var,
            prompt_xs=prompt_xs,
            prompt_ys=prompt_ys,
            x_generator=x_generator,
            print_progress=print_progress,
        )
    )(keys)


@functools.partial(
    jax.jit,
    static_argnames=[
        "num_generations_per_prompt",
        "num_examples",
        "x_generator",
        "print_progress",
    ],
)
def generate_multi_prompt(
    model: InContextRegressionTransformer,
    key: PRNGKeyArray,
    *,
    prompt_xs: Float[Array, "num_prompts prompt_len task_dim"],
    prompt_ys: Float[Array, "num_prompts prompt_len"],
    num_generations_per_prompt: int,
    num_examples: int,
    noise_var: float,
    x_generator: XGenerator = jax.random.normal,
    print_progress: bool = False,
) -> tuple[
    Float[Array, "num_prompts num_generations num_examples task_dim"],
    Float[Array, "num_prompts num_generations num_examples"],
]:
    """
    Generate N sequences from M prompts. Vmaps generate_batch.

    Args:
        model: The transformer model
        key: PRNG key for sampling
        prompt_xs: Prompt inputs [num_prompts, prompt_len, task_dim]
        prompt_ys: Prompt outputs [num_prompts, prompt_len]
        num_generations_per_prompt: Number of sequences to generate per prompt
        num_examples: Length of each sequence
        noise_var: Noise variance for gaussian/point heads
        x_generator: Function to generate random x values (default: jax.random.normal)

    Returns:
        (xs, ys): Generated input-output pairs
            xs: [num_prompts, num_generations_per_prompt, num_examples, task_dim]
            ys: [num_prompts, num_generations_per_prompt, num_examples]
    """
    num_prompts = prompt_xs.shape[0]
    keys = jax.random.split(key, num_prompts)
    return jax.vmap(
        lambda k, px, py: generate_batch(
            model,
            k,
            num_generations=num_generations_per_prompt,
            num_examples=num_examples,
            noise_var=noise_var,
            prompt_xs=px,
            prompt_ys=py,
            x_generator=x_generator,
            print_progress=print_progress,
        )
    )(keys, prompt_xs, prompt_ys)
