from __future__ import annotations

import functools
from typing import Literal, Self

import einops
import jax
import jax.numpy as jnp
import optax
import strux
from jaxtyping import Array, Float, PRNGKeyArray


GAUSSIAN_VAR_EPS = 1e-6


@jax.jit
def gaussian_nll(
    y: Float[Array, "..."],
    mean: Float[Array, "..."],
    var: Float[Array, "..."],
) -> Float[Array, "..."]:
    return 0.5 * (jnp.log(2 * jnp.pi * var) + jnp.square(y - mean) / var)


@jax.jit
def variance_over_noise(
    raw_scale: Float[Array, "..."],
    noise_var: float,
) -> Float[Array, "..."]:
    return noise_var + jax.nn.softplus(raw_scale) + GAUSSIAN_VAR_EPS


@jax.jit
def mog_nll(
    y: Float[Array, "..."],
    logits: Float[Array, "... K"],
    means: Float[Array, "... K"],
    vars: Float[Array, "... K"],
) -> Float[Array, "..."]:
    """Mixture of Gaussians negative log-likelihood with stable logsumexp."""
    log_pi = jax.nn.log_softmax(logits, axis=-1)
    component_nlls = gaussian_nll(y[..., None], means, vars)
    log_probs = log_pi - component_nlls
    return -jax.scipy.special.logsumexp(log_probs, axis=-1)


# # #
# Architecture


@strux.struct
class LinearTransform:
    weights: Float[Array, "num_inputs num_outputs"]

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_inputs", "num_outputs"])
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_outputs: int,
    ) -> Self:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        weights = jax.random.uniform(
            key=key,
            shape=(num_inputs, num_outputs),
            minval=-bound,
            maxval=+bound,
        )
        return LinearTransform(weights=weights)

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ self.weights


@strux.struct
class AffineTransform:
    weights: Float[Array, "num_inputs num_outputs"]
    biases: Float[Array, "num_outputs"]

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_inputs", "num_outputs"])
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_outputs: int,
    ) -> Self:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        weights = jax.random.uniform(
            key=key,
            shape=(num_inputs, num_outputs),
            minval=-bound,
            maxval=+bound,
        )
        biases = jnp.zeros(num_outputs)
        return AffineTransform(weights=weights, biases=biases)

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ self.weights + self.biases


@functools.partial(strux.struct, static_fieldnames=["num_heads"])
class MultiHeadedCausalSelfAttention:
    QKV: LinearTransform["3"]
    output_transform: LinearTransform
    num_heads: int

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["embed_size", "num_heads"])
    def init(
        key: PRNGKeyArray,
        embed_size: int,
        num_heads: int,
    ) -> Self:
        key_qkv, key = jax.random.split(key)
        QKV = jax.vmap(
            LinearTransform.init,
            in_axes=(0, None, None),
        )(
            jax.random.split(key_qkv, 3),
            embed_size,
            embed_size,
        )
        key_out, key = jax.random.split(key)
        output_transform = LinearTransform.init(
            key_out,
            embed_size,
            embed_size,
        )
        return MultiHeadedCausalSelfAttention(
            QKV=QKV,
            output_transform=output_transform,
            num_heads=num_heads,
        )

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        # perform query, key, value transformations (on all heads at once)
        qkv = jax.vmap(
            type(self.QKV).forward,  # two-argument version of self.QKV.forward
            in_axes=(0, None),
        )(self.QKV, x)

        # reshape the embed dimension into separate heads
        qkv_perhead = einops.rearrange(
            qkv,
            "qkv t (num_heads head_size) -> qkv t num_heads head_size",
            num_heads=self.num_heads,
        )

        # vmap the attention computation across each head
        def single_head_attention(
            qkv: Float[Array, "3 t head_size"],
        ) -> Float[Array, "t head_size"]:
            q, k, v = qkv
            t, head_size = q.shape
            # compute raw affinities                tq c @ c tk -> tq tk
            a = q @ k.T
            # scale                                 tq tk / . . -> tq tk
            a = a * jax.lax.rsqrt(jnp.float32(head_size))
            # apply causal mask                     tq tk + t t -> tq tk
            a = jnp.where(
                jnp.tril(jnp.ones((t, t), dtype=bool)),  # lower triangular mask
                a,
                -jnp.inf,
            )
            # convert affinities to mixing weights  tq tk -> tq prob(tk)
            p = jax.nn.softmax(a, axis=-1)
            # mix values for each key               tq prob(tk) @ tv c -> t c
            y = p @ v
            return y

        y_perhead = jax.vmap(
            single_head_attention,
            in_axes=2,  # qkv t vmap(num_heads) head_size
            out_axes=1,  # t vmap(num_heads) head_size
        )(qkv_perhead)

        # recombine heads into new embedding dimension
        y = einops.rearrange(
            y_perhead,
            "t num_heads head_size -> t (num_heads head_size)",
        )

        # for each token, project back into residual stream
        y_ = jax.vmap(self.output_transform.forward)(y)

        return y_


@strux.struct
class MLP:
    layer1: AffineTransform
    layer2: AffineTransform

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=["num_inputs", "num_hidden", "num_outputs"],
    )
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
    ) -> Self:
        k1, k2 = jax.random.split(key)
        layer1 = AffineTransform.init(k1, num_inputs, num_hidden)
        layer2 = AffineTransform.init(k2, num_hidden, num_outputs)
        return MLP(layer1=layer1, layer2=layer2)

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        x = self.layer1.forward(x)
        x = jax.nn.relu(x)
        x = self.layer2.forward(x)
        return x


@strux.struct
class LayerNorm:
    loc: Float[Array, "size"]
    scale: Float[Array, "size"]

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["size"])
    def init(
        key: PRNGKeyArray,
        size: int,
    ) -> Self:
        return LayerNorm(
            loc=jnp.zeros(size),
            scale=jnp.ones(size),
        )

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "size"],
    ) -> Float[Array, "size"]:
        x_mean = jnp.mean(x)
        x_rstd = jax.lax.rsqrt(jnp.var(x) + 1e-5)
        x_norm = (x - x_mean) * x_rstd
        return x_norm * self.scale + self.loc


@strux.struct
class DecodeTransformerBlock:
    layernorm1: LayerNorm
    attention: MultiHeadedCausalSelfAttention
    layernorm2: LayerNorm
    compute: MLP

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=["embed_size", "num_heads", "mlp_size"],
    )
    def init(
        key: PRNGKeyArray,
        embed_size: int,
        num_heads: int,
        mlp_size: int,
    ) -> Self:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        layernorm1 = LayerNorm.init(key=k1, size=embed_size)
        attention = MultiHeadedCausalSelfAttention.init(
            key=k2,
            embed_size=embed_size,
            num_heads=num_heads,
        )
        layernorm2 = LayerNorm.init(key=k3, size=embed_size)
        compute = MLP.init(
            key=k4,
            num_inputs=embed_size,
            num_hidden=mlp_size,
            num_outputs=embed_size,
        )
        return DecodeTransformerBlock(
            layernorm1=layernorm1,
            attention=attention,
            layernorm2=layernorm2,
            compute=compute,
        )

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        # pre layer norm (per-token)
        x_norm = jax.vmap(self.layernorm1.forward)(x)
        # attention (between tokens, residual)
        x = x + self.attention.forward(x_norm)
        # pre layer norm (per-token)
        x_norm = jax.vmap(self.layernorm2.forward)(x)
        # compute (per-token, residual)
        x = x + jax.vmap(self.compute.forward)(x_norm)
        return x


@strux.struct
class DecodeTransformer:
    token_embedding: LinearTransform
    postn_embedding: LinearTransform
    blocks: DecodeTransformerBlock["num_blocks"]
    unembedding_layernorm: LayerNorm
    unembedding: AffineTransform

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=[
            "num_inputs",
            "max_context_length",
            "num_blocks",
            "num_heads",
            "embed_size",
            "mlp_size",
            "num_outputs",
        ],
    )
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        max_context_length: int,
        num_blocks: int,
        num_heads: int,
        embed_size: int,
        mlp_size: int,
        num_outputs: int,
    ) -> Self:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        # embeddings
        token_embedding = LinearTransform.init(
            k1,
            num_inputs,
            embed_size,
        )
        postn_embedding = LinearTransform.init(
            k2,
            max_context_length,
            embed_size,
        )

        # transformer blocks
        blocks = jax.vmap(
            DecodeTransformerBlock.init,
            in_axes=(0, None, None, None),
        )(
            jax.random.split(k3, num_blocks),
            embed_size,
            num_heads,
            mlp_size,
        )

        # unembedding
        unembedding_layernorm = LayerNorm.init(
            k4,
            embed_size,
        )
        unembedding = AffineTransform.init(
            k5,
            embed_size,
            num_outputs,
        )
        return DecodeTransformer(
            token_embedding=token_embedding,
            postn_embedding=postn_embedding,
            blocks=blocks,
            unembedding_layernorm=unembedding_layernorm,
            unembedding=unembedding,
        )

    @jax.jit
    def forward(
        self: Self,
        ts: Float[Array, "t num_inputs"],
    ) -> Float[Array, "t num_outputs"]:
        context_length, _num_inputs = ts.shape

        # embedding: semantic and positional token embeddings
        x_semantic = jax.vmap(self.token_embedding.forward)(ts)
        x_position = self.postn_embedding.weights[:context_length, :]
        x = x_semantic + x_position  # -> t embed_size
        # apply the num_blocks attention blocks in sequence
        x, _ = jax.lax.scan(
            lambda x, block: (block.forward(x), None),
            x,
            self.blocks,
        )  # -> t embed_size
        # unembedding: transform back to predicted next token probs
        x_norm = jax.vmap(self.unembedding_layernorm.forward)(x)
        x = jax.vmap(self.unembedding.forward)(x_norm)  # -> t num_outputs
        return x


@functools.partial(strux.struct, static_fieldnames=["head_type", "num_components"])
class InContextRegressionTransformer:
    transformer: DecodeTransformer
    head_type: Literal["point", "gaussian", "mog"]
    num_components: int

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=[
            "task_dim",
            "num_examples",
            "head_type",
            "num_components",
            "embed_size",
            "num_blocks",
            "num_heads",
            "mlp_size",
        ],
    )
    def init(
        key: PRNGKeyArray,
        task_dim: int,
        num_examples: int,
        num_blocks: int,
        num_heads: int,
        embed_size: int,
        mlp_size: int,
        head_type: Literal["point", "gaussian", "mog"] = "gaussian",
        num_components: int = 4,
    ) -> Self:
        if head_type == "point":
            num_outputs = 1
        elif head_type == "gaussian":
            num_outputs = 2
        elif head_type == "mog":
            num_outputs = 3 * num_components  # K logits + K means + K raw_scales
        else:
            raise ValueError(f"Unknown head_type: {head_type!r}")
        transformer = DecodeTransformer.init(
            key=key,
            num_inputs=task_dim + 1,  # (task_dim for x) + (1 for y)
            max_context_length=2 * num_examples,  # (1 for x) + (1 for y) per eg
            num_blocks=num_blocks,
            num_heads=num_heads,
            embed_size=embed_size,
            mlp_size=mlp_size,
            num_outputs=num_outputs,
        )
        return InContextRegressionTransformer(
            transformer=transformer,
            head_type=head_type,
            num_components=num_components,
        )

    @jax.jit
    def _forward_tokens(
        self: Self,
        xs: Float[Array, "n task_dim"],
        ys: Float[Array, "n"],
    ) -> Float[Array, "two_n num_outputs"]:
        n, task_dim = xs.shape
        toks = jnp.zeros((2 * n, task_dim + 1))
        toks = toks.at[0::2, 1:].set(xs)
        toks = toks.at[1::2, 0].set(ys)
        return self.transformer.forward(toks)

    @jax.jit
    def forward_point(
        self: Self,
        xs: Float[Array, "n task_dim"],
        ys: Float[Array, "n"],
    ) -> Float[Array, "n"]:
        preds = self._forward_tokens(xs, ys)
        return preds[0::2, 0]

    @jax.jit
    def forward_gaussian(
        self: Self,
        xs: Float[Array, "n task_dim"],
        ys: Float[Array, "n"],
        noise_var: float,
    ) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
        preds = self._forward_tokens(xs, ys)
        mean = preds[0::2, 0]
        raw_scale = preds[0::2, 1]
        var = variance_over_noise(raw_scale, noise_var=noise_var)
        return mean, var

    @jax.jit
    def forward_mog(
        self: Self,
        xs: Float[Array, "n task_dim"],
        ys: Float[Array, "n"],
    ) -> tuple[
        Float[Array, "n K"],
        Float[Array, "n K"],
        Float[Array, "n K"],
    ]:
        preds = self._forward_tokens(xs, ys)
        K = self.num_components
        logits = preds[0::2, :K]
        means = preds[0::2, K : 2 * K]
        vars = jax.nn.softplus(preds[0::2, 2 * K :]) + GAUSSIAN_VAR_EPS
        return logits, means, vars

    @jax.jit
    def forward_point_batch(
        self: Self,
        xss: Float[Array, "batch_size n task_dim"],
        yss: Float[Array, "batch_size n"],
    ) -> Float[Array, "batch_size n"]:
        return jax.vmap(self.forward_point)(xss, yss)

    @jax.jit
    def forward_gaussian_batch(
        self: Self,
        xss: Float[Array, "batch_size n task_dim"],
        yss: Float[Array, "batch_size n"],
        noise_var: float,
    ) -> tuple[Float[Array, "batch_size n"], Float[Array, "batch_size n"]]:
        return jax.vmap(self.forward_gaussian, in_axes=(0, 0, None))(
            xss, yss, noise_var
        )

    @jax.jit
    def forward_mog_batch(
        self: Self,
        xss: Float[Array, "batch_size n task_dim"],
        yss: Float[Array, "batch_size n"],
    ) -> tuple[
        Float[Array, "batch_size n K"],
        Float[Array, "batch_size n K"],
        Float[Array, "batch_size n K"],
    ]:
        return jax.vmap(self.forward_mog)(xss, yss)


# # #
# Task distributions


@strux.struct
class DiscreteTaskDistribution:
    tasks: Float[Array, "num_tasks task_dim"]

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["task_dim", "num_tasks"])
    def init(
        key: PRNGKeyArray,
        task_dim: int,
        num_tasks: int,
    ) -> Self:
        return DiscreteTaskDistribution(
            tasks=jax.random.normal(
                key=key,
                shape=(num_tasks, task_dim),
            ),
        )

    @functools.partial(
        jax.jit,
        static_argnames=[
            "batch_size",
        ],
    )
    def sample(
        self,
        key: PRNGKeyArray,
        batch_size: int,
    ) -> Float[Array, "batch_size task_dim"]:
        return jax.random.choice(
            key=key,
            a=self.tasks,
            shape=(batch_size,),
            replace=True,
        )

    @property
    def task_dim(self) -> int:
        _num_tasks, task_dim = self.tasks.shape
        return task_dim

    @property
    def num_tasks(self) -> int:
        num_tasks, _task_size = self.tasks.shape
        return num_tasks


@functools.partial(strux.struct, static_fieldnames=["task_dim"])
class GaussianTaskDistribution:
    task_dim: int

    @functools.partial(
        jax.jit,
        static_argnames=[
            "batch_size",
        ],
    )
    def sample(
        self,
        key: PRNGKeyArray,
        batch_size: int,
    ) -> Float[Array, "batch_size task_dim"]:
        return jax.random.normal(
            key=key,
            shape=(batch_size, self.task_dim),
        )


# # #
# Data distribution


@strux.struct
class RegressionSequenceDistribution:
    task_distribution: GaussianTaskDistribution | DiscreteTaskDistribution
    noise_var: float

    @functools.partial(jax.jit, static_argnames=["batch_size", "num_examples"])
    def get_batch(
        self,
        key: PRNGKeyArray,
        num_examples: int,
        batch_size: int,
    ) -> tuple[
        Float[Array, "batch_size num_examples task_dim"],
        Float[Array, "batch_size num_examples"],
    ]:
        # sample a batch of random tasks
        key_tasks, key = jax.random.split(key)
        ws: Float[Array, "batch_size task_dim"]
        ws = self.task_distribution.sample(key_tasks, batch_size)

        # sample an iid sequence of inputs and noise for each task
        key_inputs, key = jax.random.split(key)
        xs: Float[Array, "batch_size num_examples task_dim"]
        xs = jax.random.normal(
            key=key_inputs,
            shape=(batch_size, num_examples, self.task_distribution.task_dim),
        )

        # compute raw outputs
        ys: Float[Array, "batch_size num_examples"]
        ys = einops.einsum(xs, ws, "batch ex task, batch task -> batch ex")

        # add iid gaussian noise
        key_noise, key = jax.random.split(key)
        noise: Float[Array, "batch_size num_examples"]
        noise = jax.random.normal(
            key=key_noise,
            shape=(batch_size, num_examples),
        ) * jnp.sqrt(self.noise_var)
        ys_ = ys + noise

        return xs, ys_


# Saving the model
@strux.struct
class TrainState:
    model: InContextRegressionTransformer
    opt_state: optax.OptState
    train_data: RegressionSequenceDistribution
    key: PRNGKeyArray
    step: int
