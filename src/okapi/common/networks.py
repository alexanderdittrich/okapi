"""Generic network building blocks. Algorithm-specific architectures live in the algorithm files."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
from flax import nnx

ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "tanh": nnx.tanh,
    "relu": nnx.relu,
    "swish": nnx.swish,
    "silu": nnx.silu,
    "elu": nnx.elu,
    "gelu": nnx.gelu,
    "mish": jax.nn.mish,
}


class MLP(nnx.Module):
    """``Linear → [LayerNorm] → activation`` hidden layers and a linear output layer.

    Args:
        in_size: Input features.
        hidden_sizes: Width of each hidden layer.
        out_size: Output features.
        activation: Key of :data:`ACTIVATIONS`.
        layer_norm: Apply LayerNorm before each hidden activation.
        out_scale: Variance scale of the output layer's initializer (``0`` gives a
            zero-initialized output); ``None`` uses the hidden initializer.
    """

    def __init__(
        self,
        in_size: int,
        hidden_sizes: Sequence[int],
        out_size: int,
        activation: str = "swish",
        layer_norm: bool = False,
        out_scale: float | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        init = nnx.initializers.lecun_uniform()
        sizes = [in_size, *hidden_sizes]
        self.hidden = nnx.List(
            [nnx.Linear(a, b, kernel_init=init, rngs=rngs) for a, b in zip(sizes[:-1], sizes[1:])]
        )
        self.norms = nnx.List(
            [nnx.LayerNorm(b, rngs=rngs) for b in hidden_sizes] if layer_norm else []
        )
        if out_scale is None:
            out_init = init
        elif out_scale == 0.0:
            out_init = nnx.initializers.zeros_init()
        else:
            out_init = nnx.initializers.variance_scaling(out_scale, "fan_avg", "uniform")
        self.out = nnx.Linear(sizes[-1], out_size, kernel_init=out_init, rngs=rngs)
        self.activation = ACTIVATIONS[activation]

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if self.norms:
                x = self.norms[i](x)
            x = self.activation(x)
        return self.out(x)
