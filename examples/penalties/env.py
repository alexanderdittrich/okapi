"""Penalty shoot-out between two torque-controlled humanoids.

The scene and humanoids are adapted from https://github.com/goncalog/ai-robotics.
One environment, two roles, trained as a curriculum:

* ``role="striker"``: kick the ball into the goal at a random ``target`` point
  (part of the observation) and stay on your feet. The keeper is passive.
* ``role="keeper"``: a frozen striker policy shoots at random targets; the keeper
  must stand ready, then intercept the ball. The keeper may dive.

Episodes are short (``episode_length`` ~ 2 s): one shot each.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

GOAL_X = 5.5
GOAL_WIDTH = 3.66
GOAL_HEIGHT = 2.0
BALL_RADIUS = 0.11
BALL_START = (0.35, 0.12)  # in front of the striker's left foot
GRAVITY = 9.81

# qpos / qvel layout: striker (7 + 21) | keeper (7 + 21) | ball (7)
STRIKER_Q, KEEPER_Q, BALL_Q = slice(0, 28), slice(28, 56), slice(56, 63)
STRIKER_V, KEEPER_V, BALL_V = slice(0, 27), slice(27, 54), slice(54, 60)
NU = 21

_MOTORS = [
    ("abdomen_y", 40),
    ("abdomen_z", 40),
    ("abdomen_x", 40),
    ("hip_x_right", 40),
    ("hip_z_right", 40),
    ("hip_y_right", 120),
    ("knee_right", 80),
    ("ankle_x_right", 20),
    ("ankle_y_right", 20),
    ("hip_x_left", 40),
    ("hip_z_left", 40),
    ("hip_y_left", 120),
    ("knee_left", 80),
    ("ankle_x_left", 20),
    ("ankle_y_left", 20),
    ("shoulder1_right", 20),
    ("shoulder2_right", 20),
    ("elbow_right", 40),
    ("shoulder1_left", 20),
    ("shoulder2_left", 20),
    ("elbow_left", 40),
]
_BODY_GEOMS = [
    "torso",
    "waist_upper",
    "head",
    "waist_lower",
    "butt",
    "thigh_right",
    "shin_right",
    "foot1_right",
    "foot2_right",
    "thigh_left",
    "shin_left",
    "foot1_left",
    "foot2_left",
    "upper_arm_right",
    "lower_arm_right",
    "hand_right",
    "upper_arm_left",
    "lower_arm_left",
    "hand_left",
]


# ── Scene ────────────────────────────────────────────────────────────────────


def _humanoid(p: str) -> str:
    """The dm_control humanoid; the keeper is mirrored to face the striker."""
    x, f = (0.0, 1) if p == "striker" else (GOAL_X, -1)
    jersey, skin = f"body_{p}", "skin"
    return f"""
    <body name="torso_{p}" pos="{x} 0 1.282" childclass="body">
      <freejoint name="root_{p}"/>
      <geom name="torso_{p}" fromto="0 -.07 0 0 .07 0" size=".07" material="{jersey}"/>
      <geom name="waist_upper_{p}" fromto="{-0.01 * f} -.06 -.12 {-0.01 * f} .06 -.12" size=".06" material="{jersey}"/>
      <body name="head_{p}" pos="0 0 .19"><geom name="head_{p}" type="sphere" size=".09" material="{skin}"/></body>
      <body name="waist_lower_{p}" pos="{0.01 * f} 0 -.26">
        <geom name="waist_lower_{p}" fromto="0 -.06 0 0 .06 0" size=".06" material="{jersey}"/>
        <joint name="abdomen_z_{p}" pos="0 0 .065" axis="0 0 {f}" range="-45 45" class="joint_big_stiff"/>
        <joint name="abdomen_y_{p}" pos="0 0 .065" axis="0 {f} 0" range="-75 30" class="joint_big"/>
        <body name="pelvis_{p}" pos="0 0 -.165">
          <joint name="abdomen_x_{p}" pos="0 0 .1" axis="1 0 0" range="-35 35" class="joint_big"/>
          <geom name="butt_{p}" fromto="{-0.02 * f} -.07 0 {-0.02 * f} .07 0" size=".09" material="{skin}"/>
          <body name="thigh_right_{p}" pos="0 -.1 -.04">
            <joint name="hip_x_right_{p}" axis="1 0 0" class="hip_x"/>
            <joint name="hip_z_right_{p}" axis="0 0 {f}" class="hip_z"/>
            <joint name="hip_y_right_{p}" axis="0 {f} 0" class="hip_y"/>
            <geom name="thigh_right_{p}" fromto="0 0 0 0 .01 -.34" class="thigh" material="{skin}"/>
            <body name="shin_right_{p}" pos="0 .01 -.4">
              <joint name="knee_right_{p}" axis="0 {-f} 0" class="knee"/>
              <geom name="shin_right_{p}" class="shin" material="{skin}"/>
              <body name="foot_right_{p}" pos="0 0 -.39">
                <joint name="ankle_y_right_{p}" axis="0 {f} 0" class="ankle_y"/>
                <joint name="ankle_x_right_{p}" axis="1 0 {0.5 * f}" class="ankle_x"/>
                <geom name="foot1_right_{p}" class="foot1_{p}" material="{skin}"/>
                <geom name="foot2_right_{p}" class="foot2_{p}" material="{skin}"/>
              </body>
            </body>
          </body>
          <body name="thigh_left_{p}" pos="0 .1 -.04">
            <joint name="hip_x_left_{p}" axis="-1 0 0" class="hip_x"/>
            <joint name="hip_z_left_{p}" axis="0 0 {-f}" class="hip_z"/>
            <joint name="hip_y_left_{p}" axis="0 {f} 0" class="hip_y"/>
            <geom name="thigh_left_{p}" fromto="0 0 0 0 -.01 -.34" class="thigh" material="{skin}"/>
            <body name="shin_left_{p}" pos="0 -.01 -.4">
              <joint name="knee_left_{p}" axis="0 {-f} 0" class="knee"/>
              <geom name="shin_left_{p}" class="shin" material="{skin}"/>
              <body name="foot_left_{p}" pos="0 0 -.39">
                <joint name="ankle_y_left_{p}" axis="0 {f} 0" class="ankle_y"/>
                <joint name="ankle_x_left_{p}" axis="-1 0 {-0.5 * f}" class="ankle_x"/>
                <geom name="foot1_left_{p}" class="foot1_{p}" material="{skin}"/>
                <geom name="foot2_left_{p}" class="foot2_{p}" material="{skin}"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="upper_arm_right_{p}" pos="0 -.17 .06">
        <joint name="shoulder1_right_{p}" axis="2 {f} {f}" class="shoulder"/>
        <joint name="shoulder2_right_{p}" axis="0 {-f} {f}" class="shoulder"/>
        <geom name="upper_arm_right_{p}" fromto="0 0 0 {0.16 * f} -.16 -.16" class="arm_upper" material="{jersey}"/>
        <body name="lower_arm_right_{p}" pos="{0.18 * f} -.18 -.18">
          <joint name="elbow_right_{p}" axis="0 {-f} {f}" class="elbow"/>
          <geom name="lower_arm_right_{p}" fromto="{0.01 * f} .01 .01 {0.17 * f} .17 .17" class="arm_lower" material="{skin}"/>
          <body name="hand_right_{p}" pos="{0.18 * f} .18 .18">
            <geom name="hand_right_{p}" zaxis="1 {f} 1" class="hand" material="{skin}"/>
          </body>
        </body>
      </body>
      <body name="upper_arm_left_{p}" pos="0 .17 .06">
        <joint name="shoulder1_left_{p}" axis="-2 {f} {-f}" class="shoulder"/>
        <joint name="shoulder2_left_{p}" axis="0 {-f} {-f}" class="shoulder"/>
        <geom name="upper_arm_left_{p}" fromto="0 0 0 {0.16 * f} .16 -.16" class="arm_upper" material="{jersey}"/>
        <body name="lower_arm_left_{p}" pos="{0.18 * f} .18 -.18">
          <joint name="elbow_left_{p}" axis="0 {-f} {-f}" class="elbow"/>
          <geom name="lower_arm_left_{p}" fromto="{0.01 * f} -.01 .01 {0.17 * f} -.17 .17" class="arm_lower" material="{skin}"/>
          <body name="hand_left_{p}" pos="{0.18 * f} -.18 .18">
            <geom name="hand_left_{p}" zaxis="1 {-f} 1" class="hand" material="{skin}"/>
          </body>
        </body>
      </body>
    </body>"""


def _contacts() -> str:
    pairs = []
    for p in ("striker", "keeper"):
        pairs += [
            f'<exclude body1="waist_lower_{p}" body2="thigh_right_{p}"/>',
            f'<exclude body1="waist_lower_{p}" body2="thigh_left_{p}"/>',
        ]
        for side in ("left", "right"):  # feet on the floor, both feet and shins on the ball
            pairs += [
                f'<pair geom1="foot1_{side}_{p}" geom2="floor"/>',
                f'<pair geom1="foot2_{side}_{p}" geom2="floor"/>',
            ]
    for g in ["foot1_left", "foot2_left", "shin_left", "foot1_right", "foot2_right", "shin_right"]:
        pairs.append(f'<pair geom1="{g}_striker" geom2="ball"/>')
    for g in _BODY_GEOMS:  # the keeper blocks with any body part; both can land on the floor
        pairs.append(f'<pair geom1="{g}_keeper" geom2="ball"/>')
        if not g.startswith("foot"):
            pairs.append(f'<pair geom1="{g}_keeper" geom2="floor"/>')
    for g in ["torso", "head", "butt", "thigh_right", "thigh_left", "hand_right", "hand_left"]:
        pairs.append(f'<pair geom1="{g}_striker" geom2="floor"/>')
    pairs += [
        '<pair geom1="post_left" geom2="ball"/>',
        '<pair geom1="post_right" geom2="ball"/>',
        '<pair geom1="crossbar" geom2="ball"/>',
    ]
    return "\n    ".join(pairs)


def make_xml() -> str:
    motors = "\n    ".join(
        f'<motor name="{n}_{p}" gear="{g}" joint="{n}_{p}"/>'
        for p in ("striker", "keeper")
        for n, g in _MOTORS
    )
    return f"""<mujoco model="penalties">
  <option timestep="0.005" iterations="6" ls_iterations="6" solver="CG">
    <flag eulerdamp="disable"/>
  </option>
  <visual>
    <map force="0.1" zfar="50" znear="0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="1280" offheight="720" elevation="-20" azimuth="120"/>
    <quality shadowsize="4096" offsamples="8"/>
  </visual>
  <statistic center="{GOAL_X / 2} 0 1" extent="8"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2=".1 .2 .4" width="32" height="512"/>
    <material name="body_striker" rgba="0.1 0.2 0.8 1"/>
    <material name="body_keeper" rgba="0.85 0.6 0.05 1"/>
    <material name="skin" rgba="0.7 0.5 0.3 1"/>
    <texture name="pitch" type="2d" builtin="checker" rgb1=".24 .48 .14" rgb2=".30 .55 .18" width="512" height="512"/>
    <material name="pitch" reflectance="0.05" texture="pitch" texrepeat="6 1" texuniform="true"/>
    <texture name="ball" type="cube" builtin="checker" rgb1="0.1 0.1 0.1" rgb2="0.95 0.95 0.95" width="64" height="64"/>
    <material name="ball" texture="ball" rgba="1 1 1 1"/>
    <material name="white" rgba="0.95 0.95 0.95 1"/>
  </asset>
  <default>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
    <default class="body">
      <geom type="capsule" condim="3" friction=".7" solimp=".9 .99 .003" solref=".015 1" contype="0" conaffinity="0"/>
      <joint type="hinge" damping=".2" stiffness="1" armature=".01" limited="true" solimplimit="0 .99 .01"/>
      <default class="thigh"><geom size=".06"/></default>
      <default class="shin"><geom fromto="0 0 0 0 0 -.3" size=".049"/></default>
      <default class="foot1_striker"><geom size=".027" fromto="-.07 -.01 0 .14 -.03 0"/></default>
      <default class="foot2_striker"><geom size=".027" fromto="-.07 .01 0 .14 .03 0"/></default>
      <default class="foot1_keeper"><geom size=".027" fromto="-.14 -.03 0 .07 -.01 0"/></default>
      <default class="foot2_keeper"><geom size=".027" fromto="-.14 .03 0 .07 .01 0"/></default>
      <default class="arm_upper"><geom size=".04"/></default>
      <default class="arm_lower"><geom size=".031"/></default>
      <default class="hand"><geom type="sphere" size=".04"/></default>
      <default class="joint_big">
        <joint damping="5" stiffness="10"/>
        <default class="hip_x"><joint range="-30 10"/></default>
        <default class="hip_z"><joint range="-60 35"/></default>
        <default class="hip_y"><joint axis="0 1 0" range="-150 20"/></default>
        <default class="joint_big_stiff"><joint stiffness="20"/></default>
      </default>
      <default class="knee"><joint pos="0 0 .02" axis="0 -1 0" range="-160 2"/></default>
      <default class="ankle_y"><joint pos="0 0 .08" axis="0 1 0" stiffness="6" range="-50 50"/></default>
      <default class="ankle_x"><joint pos="0 0 .04" stiffness="3" range="-50 50"/></default>
      <default class="shoulder"><joint range="-85 60"/></default>
      <default class="elbow"><joint range="-100 50" stiffness="0"/></default>
    </default>
  </default>
  <worldbody>
    <light directional="true" diffuse=".8 .8 .8" pos="{GOAL_X / 2} 0 15" dir="0 0 -10"/>
    <geom name="floor" type="plane" size="{GOAL_X * 2} 8 .5" material="pitch" contype="1" conaffinity="1"/>
    <camera name="broadcast" pos="{GOAL_X / 2 + 0.3} -4.6 1.9" xyaxes="1 0 0 0 0.35 1" fovy="60" mode="fixed"/>
    <camera name="behind" pos="-3.2 -1.2 2.2" xyaxes="0.35 -0.94 0 0.33 0.12 0.94" mode="fixed"/>
    {_humanoid("striker")}
    {_humanoid("keeper")}
    <geom name="post_left" type="cylinder" size="0.06 {GOAL_HEIGHT / 2}" pos="{GOAL_X} {-GOAL_WIDTH / 2} {GOAL_HEIGHT / 2}" material="white" contype="0" conaffinity="0"/>
    <geom name="post_right" type="cylinder" size="0.06 {GOAL_HEIGHT / 2}" pos="{GOAL_X} {GOAL_WIDTH / 2} {GOAL_HEIGHT / 2}" material="white" contype="0" conaffinity="0"/>
    <geom name="crossbar" type="cylinder" size="0.06 {GOAL_WIDTH / 2}" pos="{GOAL_X} 0 {GOAL_HEIGHT}" euler="90 0 0" material="white" contype="0" conaffinity="0"/>
    <body name="ball" pos="{BALL_START[0]} {BALL_START[1]} {BALL_RADIUS}">
      <freejoint name="ball"/>
      <geom name="ball" type="sphere" size="{BALL_RADIUS}" mass="0.43" material="ball"
            friction="0.7 0.01 0.005" solref="0.02 1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
  <contact>
    {_contacts()}
  </contact>
  <actuator>
    {motors}
  </actuator>
</mujoco>"""


# ── Environment ──────────────────────────────────────────────────────────────


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.025,  # 40 Hz control
        sim_dt=0.005,
        episode_length=80,  # 2 s: one shot
        impl="jax",
        reward_config=config_dict.create(
            upright=0.5,  # stay on your feet (striker always, keeper until the shot)
            approach=0.5,  # striker: foot to ball before the kick
            shot=5.0,  # striker: ball speed towards the target
            goal=50.0,  # striker: + / keeper: - on a goal
            accuracy=50.0,  # striker: extra goal reward for hitting the target
            intercept=2.0,  # keeper: hands near the ball's predicted crossing point
            save=50.0,  # keeper: the ball is deflected or stopped
            ctrl=0.01,
            fall=5.0,  # striker: penalty for falling (ends the episode)
        ),
    )


Policy = Callable[[jax.Array], jax.Array]


class PenaltyKick(mjx_env.MjxEnv):
    """Penalty kick for ``role`` ``"striker"`` or ``"keeper"``.

    Args:
        role: Which player is controlled by the agent.
        striker_policy: For ``role="keeper"``: ``striker_policy(striker_obs) -> action``.
    """

    def __init__(
        self,
        role: str = "striker",
        striker_policy: Policy | None = None,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        super().__init__(config or default_config(), config_overrides)
        if role == "keeper" and striker_policy is None:
            raise ValueError("role='keeper' needs a striker_policy")
        self.role = role
        self._striker_policy = striker_policy
        self._mj_model = mujoco.MjModel.from_xml_string(make_xml())
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        body = lambda n: self._mj_model.body(n).id
        self._torso = {p: body(f"torso_{p}") for p in ("striker", "keeper")}
        self._feet = jnp.array([body("foot_left_striker"), body("foot_right_striker")])
        self._hands = jnp.array(
            [
                body("hand_left_keeper"),
                body("hand_right_keeper"),
                body("head_keeper"),
                body("torso_keeper"),
            ]
        )

    # ── Reset / step ─────────────────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, q_key, v_key, target_key = jax.random.split(rng, 4)
        qpos = jnp.array(self._mj_model.qpos0)
        joints = jnp.r_[7:28, 35:56]
        qpos = qpos.at[joints].add(jax.random.uniform(q_key, (42,), minval=-0.02, maxval=0.02))
        qvel = (
            jnp.zeros(self._mj_model.nv)
            .at[:54]
            .set(jax.random.uniform(v_key, (54,), minval=-0.02, maxval=0.02))
        )
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel, impl=self._config.impl)
        data = mjx.forward(self._mjx_model, data)
        lo = jnp.array([-GOAL_WIDTH / 2 + 0.35, 0.25])
        hi = jnp.array([GOAL_WIDTH / 2 - 0.35, GOAL_HEIGHT - 0.35])
        info = {
            "rng": rng,
            "target": jax.random.uniform(target_key, (2,), minval=lo, maxval=hi),
            "step": jnp.zeros((), jnp.int32),
            "shot_over": jnp.zeros(()),  # ball crossed the goal line or was saved
            "kicked": jnp.zeros(()),  # ball has been struck
        }
        metrics = {k: jnp.zeros(()) for k in ("goal", "save", "upright", "shot", "intercept")}
        return mjx_env.State(
            data, self._obs(data, info), jnp.zeros(()), jnp.zeros(()), metrics, info
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        if self.role == "striker":  # the passive keeper holds its standing pose (PD)
            q, qd = state.data.qpos[KEEPER_Q][7:], state.data.qvel[KEEPER_V][6:]
            ctrl = jnp.concatenate([action, jnp.clip(-2.0 * q - 0.2 * qd, -1.0, 1.0)])
        else:
            striker_action = self._striker_policy(state.obs["striker"])
            ctrl = jnp.concatenate([jax.lax.stop_gradient(striker_action), action])
        data = self._physics_step(state.data, ctrl)
        unstable = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()

        ball0, ball = state.data.qpos[BALL_Q][:3], data.qpos[BALL_Q][:3]
        vel0, vel = state.data.qvel[BALL_V][:3], data.qvel[BALL_V][:3]
        crossed = (ball0[0] < GOAL_X) & (ball[0] >= GOAL_X) & (info["shot_over"] == 0)
        in_frame = (jnp.abs(ball[1]) < GOAL_WIDTH / 2 - BALL_RADIUS) & (
            ball[2] < GOAL_HEIGHT - BALL_RADIUS
        )
        goal = (crossed & in_frame).astype(jnp.float32)
        info["kicked"] = jnp.maximum(info["kicked"], (vel[0] > 2.0).astype(jnp.float32))
        # a save: a struck ball near the keeper changes velocity abruptly without a goal
        near_keeper = ball[0] > GOAL_X - 1.5
        deflected = (jnp.linalg.norm(vel - vel0) > 2.5) & near_keeper & (info["kicked"] > 0)
        save = (deflected & (info["shot_over"] == 0) & ~crossed).astype(jnp.float32)

        if self.role == "striker":
            reward, done, terms = self._striker_reward(data, action, info, goal)
        else:
            reward, done, terms = self._keeper_reward(data, action, info, goal, save)
        info["shot_over"] = jnp.maximum(info["shot_over"], jnp.maximum(goal, save))
        info["shot_over"] = jnp.maximum(info["shot_over"], crossed.astype(jnp.float32))
        info["step"] = info["step"] + 1
        metrics = {
            k: jnp.where(unstable, 0.0, v) for k, v in {**terms, "goal": goal, "save": save}.items()
        }
        # a diverged simulation ends the episode; the auto-reset discards its state
        obs = jax.tree.map(lambda x: jnp.where(unstable, 0.0, x), self._obs(data, info))
        reward = jnp.where(unstable, 0.0, reward)
        done = jnp.maximum(done, unstable.astype(jnp.float32))
        return state.replace(
            data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info
        )

    def _physics_step(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        """``n_substeps`` physics steps, clamping velocities after each one for stability."""

        def substep(data, _):
            data = mjx.step(self._mjx_model, data.replace(ctrl=ctrl))
            return data.replace(qvel=jnp.clip(data.qvel, -100.0, 100.0)), None

        return jax.lax.scan(substep, data, None, self.n_substeps)[0]

    # ── Rewards ──────────────────────────────────────────────────────────────

    def _upright(self, data: mjx.Data, player: str) -> jax.Array:
        torso = data.xpos[self._torso[player]]
        height = jnp.clip((torso[2] - 0.8) / 0.4, 0.0, 1.0)
        up = data.xmat[self._torso[player]].reshape(3, 3)[2, 2]  # torso z-axis vs. world z
        return height * jnp.clip(up, 0.0, 1.0)

    def _striker_reward(self, data, action, info, goal):
        rc = self._config.reward_config
        ball, vel = data.qpos[BALL_Q][:3], data.qvel[BALL_V][:3]
        upright = self._upright(data, "striker")
        target = jnp.array([GOAL_X, info["target"][0], info["target"][1]])
        direction = (target - ball) / (jnp.linalg.norm(target - ball) + 1e-6)
        in_play = (info["shot_over"] == 0) & (ball[0] < GOAL_X)
        shot = jnp.clip(jnp.dot(vel, direction), 0.0, 15.0) / 15.0 * in_play
        foot_dist = jnp.min(jnp.linalg.norm(data.xpos[self._feet] - ball, axis=-1))
        approach = jnp.exp(-jnp.square(foot_dist) / 0.05) * (1.0 - info["kicked"])
        miss = jnp.linalg.norm(ball[1:] - info["target"])
        accuracy = goal * jnp.exp(-jnp.square(miss) / 0.25)
        fallen = data.xpos[self._torso["striker"]][2] < 0.75
        reward = (
            rc.upright * upright
            + rc.approach * approach
            + rc.shot * shot
            + rc.goal * goal
            + rc.accuracy * accuracy
            - rc.fall * fallen
            - rc.ctrl * jnp.sum(jnp.square(action))
        )
        terms = {"upright": upright, "shot": shot, "intercept": jnp.zeros(())}
        return reward, fallen.astype(jnp.float32), terms

    def _keeper_reward(self, data, action, info, goal, save):
        rc = self._config.reward_config
        ball, vel = data.qpos[BALL_Q][:3], data.qvel[BALL_V][:3]
        upright = self._upright(data, "keeper") * (1.0 - info["kicked"])
        # predicted point where the ball crosses the keeper's plane (ballistic flight)
        plane = GOAL_X - 0.3
        t = jnp.clip((plane - ball[0]) / jnp.maximum(vel[0], 0.5), 0.0, 2.0)
        cross = jnp.array(
            [
                plane,
                ball[1] + vel[1] * t,
                jnp.maximum(ball[2] + vel[2] * t - 0.5 * GRAVITY * t**2, BALL_RADIUS),
            ]
        )
        reach = jnp.min(jnp.linalg.norm(data.xpos[self._hands] - cross, axis=-1))
        incoming = (info["kicked"] > 0) & (info["shot_over"] == 0) & (vel[0] > 0.5)
        intercept = jnp.exp(-jnp.square(reach) / 0.3) * incoming
        reward = (
            rc.upright * upright
            + rc.intercept * intercept
            + rc.save * save
            - rc.goal * goal
            - rc.ctrl * jnp.sum(jnp.square(action))
        )
        terms = {"upright": upright, "shot": jnp.zeros(()), "intercept": intercept}
        return reward, jnp.zeros(()), terms

    # ── Observations ─────────────────────────────────────────────────────────

    def _player_obs(self, data: mjx.Data, player: str, q: slice, v: slice) -> jax.Array:
        torso = data.xpos[self._torso[player]]
        ball, ball_vel = data.qpos[BALL_Q][:3], data.qvel[BALL_V][:3]
        return jnp.concatenate(
            [
                data.qpos[q][2:],  # torso height, orientation, joint angles (26)
                data.qvel[v],  # (27)
                ball - torso,
                ball_vel,  # (6)
            ]
        )

    def _obs(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, jax.Array]:
        time = info["step"].astype(jnp.float32)[None] / self._config.episode_length
        striker = jnp.concatenate(
            [
                self._player_obs(data, "striker", STRIKER_Q, STRIKER_V),
                (data.xpos[self._feet] - data.qpos[BALL_Q][:3]).ravel(),  # feet relative to ball
                info["target"],
                time,
            ]
        )  # 68
        keeper = jnp.concatenate(
            [
                self._player_obs(data, "keeper", KEEPER_Q, KEEPER_V),
                (
                    data.xpos[self._feet] - data.xpos[self._torso["keeper"]]
                ).ravel(),  # striker's feet
                time,
            ]
        )  # 66
        obs = {"striker": striker, "keeper": keeper}
        obs["state"] = obs[self.role]
        return obs

    # ── MjxEnv interface ─────────────────────────────────────────────────────

    @property
    def xml_path(self) -> str:
        return ""

    @property
    def action_size(self) -> int:
        return NU

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
