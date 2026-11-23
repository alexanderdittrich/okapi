# API reference

## Package

```{eval-rst}
.. automodule:: okapi
   :members: ALGORITHMS, get_algorithm, load_policy
```

## Algorithms

```{eval-rst}
.. autoclass:: okapi.ppo.PPOConfig
   :no-members:
.. autofunction:: okapi.ppo.train
.. autofunction:: okapi.ppo.load_policy

.. autoclass:: okapi.ppo_gru.PPOGRUConfig
   :no-members:
.. autofunction:: okapi.ppo_gru.train
.. autofunction:: okapi.ppo_gru.load_policy

.. autoclass:: okapi.sac.SACConfig
   :no-members:
.. autofunction:: okapi.sac.train
.. autofunction:: okapi.sac.load_policy

.. autoclass:: okapi.simbav2.SimbaV2Config
   :no-members:
.. autofunction:: okapi.simbav2.train
.. autofunction:: okapi.simbav2.load_policy

.. autoclass:: okapi.bro.BROConfig
   :no-members:
.. autofunction:: okapi.bro.train
.. autofunction:: okapi.bro.load_policy

.. autoclass:: okapi.dreamerv3.DreamerV3Config
   :no-members:
.. autofunction:: okapi.dreamerv3.train
.. autofunction:: okapi.dreamerv3.load_policy

.. autoclass:: okapi.tdmpc2.TDMPC2Config
   :no-members:
.. autofunction:: okapi.tdmpc2.train
.. autofunction:: okapi.tdmpc2.load_policy
```

## Common

```{eval-rst}
.. autoclass:: okapi.common.config.TrainConfig
   :no-members:
.. autoclass:: okapi.common.policy.Policy
   :members: __call__
.. automodule:: okapi.common.envs
   :members: load_env, make_env, step_or_reset, make_evaluator, obs_to_dict
.. automodule:: okapi.common.replay
   :members: init, add, sample_transitions, sample_sequences
.. automodule:: okapi.common.training
   :members: setup, train_loop
.. automodule:: okapi.common.checkpoint
   :members: save, load, steps, load_metadata
.. automodule:: okapi.common.logger
   :members: Logger
```
