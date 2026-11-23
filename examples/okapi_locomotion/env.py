"""Joystick locomotion for the okapi quadruped.

Commands: forward velocity in [0, 1.5] m/s and yaw rate in [-0.8, 0.8] rad/s.
Control: joint position targets around a standing pose (PD in MuJoCo).

Smooth gaits come from a trot clock: the phase is part of the observation and a
``feet_phase`` reward asks the diagonal leg pairs to follow a sinusoidal swing
height profile. Penalties on action rate, action jerk, vertical bounce and
body roll / pitch rate remove the remaining jitter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from scenery import add_scenery

XML_PATH = Path(__file__).resolve().parent / "okapi.xml"
FEET = ["foot_FL", "foot_FR", "foot_RL", "foot_RR"]
FOOT_RADIUS = 0.028
TROT_OFFSETS = jnp.array([0.0, 0.5, 0.5, 0.0])  # FL+RR swing together, then FR+RL


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        impl="jax",  # MJX
        ctrl_dt=0.02,  # 50 Hz control
        sim_dt=0.004,  # 250 Hz physics
        episode_length=1000,
        Kd=1.5,  # joint damping (derivative gain of the position actuators)
        action_scale=0.3,  # rad, offset from the standing pose
        soft_joint_pos_limit_factor=0.95,
        gait_frequency=2.0,  # Hz
        swing_height=0.08,  # m
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05, linvel=0.1
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,
                feet_phase=1.0,
                lin_vel_z=-2.0,
                ang_vel_xy=-0.2,
                orientation=-5.0,
                pose=0.2,
                stand_still=-1.0,
                dof_pos_limits=-1.0,
                termination=-1.0,
                torques=-0.0002,
                energy=-0.002,
                action_rate=-0.05,
                action_smoothness=-0.02,
                feet_slip=-0.2,
            ),
            tracking_sigma=0.25,
            phase_sigma=0.01,
        ),
        command_config=config_dict.create(
            a=[1.5, 0.0, 0.8],  # amplitude of vx, vy, yaw rate
            b=[0.9, 1.0, 0.7],  # probability of resampling each component
        ),
    )


class OkapiJoystick(mjx_env.MjxEnv):
    """Velocity-tracking locomotion for the okapi quadruped."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
        scenery: bool = True,
    ):
        """``scenery`` adds decorative rainforest geoms (no effect on physics or observations)."""
        super().__init__(config or default_config(), config_overrides)
        spec = mujoco.MjSpec.from_file(str(XML_PATH))
        if scenery:
            add_scenery(spec)
        self._mj_model = spec.compile()
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.dof_damping[6:] = self._config.Kd
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        m = self._mj_model

        key = m.key("stand").id
        self._init_q = jnp.array(m.key_qpos[key])
        self._default_pose = jnp.array(m.key_ctrl[key])
        lowers, uppers = m.jnt_range[1:].T
        self._soft_lowers = lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = uppers * self._config.soft_joint_pos_limit_factor
        self._torso_id = m.body("torso").id
        self._foot_site_ids = np.array([m.site(n).id for n in FEET])
        self._contact_adr = np.array([m.sensor_adr[m.sensor(f"{n}_floor_found").id] for n in FEET])
        self._foot_vel_adr = np.array(
            [[m.sensor_adr[m.sensor(f"{n}_vel").id] + k for k in range(3)] for n in FEET]
        )
        self._cmd_a = jnp.array(self._config.command_config.a)
        self._cmd_b = jnp.array(self._config.command_config.b)

    # ── Kinematics helpers ───────────────────────────────────────────────────

    def _rotation(self, data: mjx.Data) -> jax.Array:
        return data.xmat[self._torso_id].reshape(3, 3)

    def _local_linvel(self, data: mjx.Data) -> jax.Array:
        return self._rotation(data).T @ data.sensordata[0:3]

    def _gyro(self, data: mjx.Data) -> jax.Array:
        return self._rotation(data).T @ data.sensordata[3:6]

    def _gravity(self, data: mjx.Data) -> jax.Array:
        return self._rotation(data).T @ jnp.array([0.0, 0.0, -1.0])

    def _foot_heights(self, data: mjx.Data) -> jax.Array:
        return data.site_xpos[self._foot_site_ids, 2] - FOOT_RADIUS

    def _swing_reference(self, phase: jax.Array) -> jax.Array:
        """Desired height of each foot for the trot clock ``phase`` in [0, 1)."""
        leg_phase = jnp.mod(phase + TROT_OFFSETS, 1.0)
        swing = leg_phase < 0.5
        return jnp.where(swing, self._config.swing_height * jnp.sin(2 * jnp.pi * leg_phase), 0.0)

    # ── Reset / step ─────────────────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, xy_key, yaw_key, vel_key, cmd_key, time_key, phase_key = jax.random.split(rng, 7)
        qpos = self._init_q.at[0:2].add(jax.random.uniform(xy_key, (2,), minval=-0.5, maxval=0.5))
        yaw = jax.random.uniform(yaw_key, (), minval=-jnp.pi, maxval=jnp.pi)
        qpos = qpos.at[3:7].set(jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)]))
        qvel = (
            jnp.zeros(self.mjx_model.nv)
            .at[0:6]
            .set(jax.random.uniform(vel_key, (6,), minval=-0.5, maxval=0.5))
        )
        data = mjx_env.make_data(
            self.mj_model, qpos=qpos, qvel=qvel, ctrl=self._default_pose, impl=self._config.impl
        )
        data = mjx.forward(self.mjx_model, data)

        cmd = jax.random.uniform(cmd_key, (3,), minval=-self._cmd_a, maxval=self._cmd_a)
        cmd = cmd.at[0].set(jnp.abs(cmd[0])).at[1].set(0.0)
        info = {
            "rng": rng,
            "command": cmd,
            "steps_until_next_cmd": self._hold_time(time_key),
            "phase": jax.random.uniform(phase_key, ()),
            "last_act": jnp.zeros(self.mjx_model.nu),
            "last_last_act": jnp.zeros(self.mjx_model.nu),
        }
        metrics = {f"reward/{k}": jnp.zeros(()) for k in self._config.reward_config.scales}
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = state.info
        targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(self.mjx_model, state.data, targets, self.n_substeps)
        contact = data.sensordata[self._contact_adr] > 0
        info["phase"] = jnp.mod(info["phase"] + self.dt * self._config.gait_frequency, 1.0)

        done = self._get_termination(data)
        rewards = self._get_reward(data, action, info, done, contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10_000.0)

        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["steps_until_next_cmd"] -= 1
        info["rng"], cmd_key, time_key = jax.random.split(info["rng"], 3)
        resample = info["steps_until_next_cmd"] <= 0
        info["command"] = jnp.where(
            resample, self._sample_command(cmd_key, info["command"]), info["command"]
        )
        info["steps_until_next_cmd"] = jnp.where(
            resample, self._hold_time(time_key), info["steps_until_next_cmd"]
        )
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        obs = self._get_obs(data, info)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        upside_down = self._rotation(data)[2, 2] < 0.0
        too_low = data.qpos[2] < 0.15
        return upside_down | too_low

    # ── Observations ─────────────────────────────────────────────────────────

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        cfg = self._config.noise_config
        linvel, gyro, gravity = self._local_linvel(data), self._gyro(data), self._gravity(data)
        jpos, jvel = data.qpos[7:], data.qvel[6:]
        info["rng"], *keys = jax.random.split(info["rng"], 6)
        noisy = lambda x, scale, k: x + (2 * jax.random.uniform(k, x.shape) - 1) * cfg.level * scale
        clock = jnp.array(
            [jnp.cos(2 * jnp.pi * info["phase"]), jnp.sin(2 * jnp.pi * info["phase"])]
        )
        state = jnp.concatenate(
            [
                noisy(linvel, cfg.scales.linvel, keys[0]),
                noisy(gyro, cfg.scales.gyro, keys[1]),
                noisy(gravity, cfg.scales.gravity, keys[2]),
                noisy(jpos, cfg.scales.joint_pos, keys[3]) - self._default_pose,
                noisy(jvel, cfg.scales.joint_vel, keys[4]),
                info["last_act"],
                info["command"],
                clock,
            ]
        )  # 50
        privileged_state = jnp.concatenate(
            [
                state,
                linvel,
                gyro,
                gravity,
                jpos - self._default_pose,
                jvel,
                data.actuator_force,
                data.sensordata[self._foot_vel_adr].ravel(),
                self._foot_heights(data),
            ]
        )  # 50 + 61
        return {"state": state, "privileged_state": privileged_state}

    # ── Reward ───────────────────────────────────────────────────────────────

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        rc = self._config.reward_config
        cmd = info["command"]
        moving = jnp.linalg.norm(cmd) > 0.05
        linvel, gyro = self._local_linvel(data), self._gyro(data)
        jpos, jvel, force = data.qpos[7:], data.qvel[6:], data.actuator_force
        foot_vel_xy = data.sensordata[self._foot_vel_adr][:, :2]

        height_error = self._foot_heights(data) - self._swing_reference(info["phase"])
        feet_phase = jnp.exp(-jnp.sum(jnp.square(height_error)) / rc.phase_sigma) * moving
        return {
            "tracking_lin_vel": jnp.exp(
                -jnp.sum(jnp.square(cmd[:2] - linvel[:2])) / rc.tracking_sigma
            ),
            "tracking_ang_vel": jnp.exp(-jnp.square(cmd[2] - gyro[2]) / rc.tracking_sigma),
            "feet_phase": feet_phase,
            "lin_vel_z": jnp.square(data.sensordata[2]),
            "ang_vel_xy": jnp.sum(jnp.square(gyro[:2])),
            "orientation": jnp.sum(jnp.square(self._gravity(data)[:2])),
            "pose": jnp.exp(-jnp.sum(jnp.square(jpos - self._default_pose))),
            "stand_still": jnp.sum(jnp.abs(jpos - self._default_pose)) * ~moving,
            "dof_pos_limits": jnp.sum(jnp.clip(self._soft_lowers - jpos, 0.0))
            + jnp.sum(jnp.clip(jpos - self._soft_uppers, 0.0)),
            "termination": done.astype(jnp.float32),
            "torques": jnp.sqrt(jnp.sum(jnp.square(force))) + jnp.sum(jnp.abs(force)),
            "energy": jnp.sum(jnp.abs(jvel * force)),
            "action_rate": jnp.sum(jnp.square(action - info["last_act"])),
            "action_smoothness": jnp.sum(
                jnp.square(action - 2 * info["last_act"] + info["last_last_act"])
            ),
            "feet_slip": jnp.sum(jnp.sum(jnp.square(foot_vel_xy), -1) * contact) * moving,
        }

    # ── Commands ─────────────────────────────────────────────────────────────

    def _hold_time(self, key: jax.Array) -> jax.Array:
        """Steps until the next command change, ~ Exp(mean 5 s)."""
        return jnp.round(jax.random.exponential(key) * 5.0 / self.dt).astype(jnp.int32)

    def _sample_command(self, key: jax.Array, cmd: jax.Array) -> jax.Array:
        new_key, resample_key, keep_key = jax.random.split(key, 3)
        new = jax.random.uniform(new_key, (3,), minval=-self._cmd_a, maxval=self._cmd_a)
        nonzero = jax.random.bernoulli(resample_key, self._cmd_b, (3,))
        change = jax.random.bernoulli(keep_key, 0.5, (3,))
        cmd = cmd - change * (cmd - new * nonzero)
        return cmd.at[0].set(jnp.abs(cmd[0])).at[1].set(0.0)

    # ── MjxEnv interface ─────────────────────────────────────────────────────

    @property
    def xml_path(self) -> str:
        return str(XML_PATH)

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
