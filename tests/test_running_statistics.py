import jax
import numpy as np

from okapi.common import running_statistics


def test_matches_numpy():
    x = jax.random.normal(jax.random.key(0), (3, 100, 4)) * 3.0 + 1.0
    state = running_statistics.init_state((4,))
    for chunk in x:
        state = running_statistics.update(state, chunk)
    flat = np.asarray(x).reshape(-1, 4)
    np.testing.assert_allclose(state.mean, flat.mean(0), rtol=1e-4)
    np.testing.assert_allclose(state.std, flat.std(0), rtol=1e-4)
