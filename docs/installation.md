# Installation

okapi uses [uv](https://docs.astral.sh/uv/) and needs Python 3.13 and an NVIDIA GPU
with a recent CUDA driver.

```bash
git clone https://github.com/alexanderdittrich/okapi.git
cd okapi
uv sync
```

Check that JAX sees the GPU:

```bash
uv run python -c "import jax; print(jax.devices())"
```

Optional dependency groups:

```bash
uv sync --group dev    # pytest, ruff
uv sync --group docs   # sphinx, to build these docs: uv run sphinx-build docs docs/_build
```

```{note}
On import, okapi sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` (so several runs can
share a GPU), `MUJOCO_GL=egl` and the XLA flag `--xla_gpu_triton_gemm_any=True`,
unless you set them yourself.
```
