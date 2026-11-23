"""Decorative rainforest scenery for the okapi: trees, undergrowth, rocks, logs and flowers.

All geoms are visual only (no mass, no collision), so they never change the physics.
"""

from __future__ import annotations

import mujoco
import numpy as np

HAZE = np.array([0.62, 0.74, 0.66])
CLEARING = (-1.5, 8.0, -3.0, 3.2)  # x_min, x_max, y_min, y_max kept free for the rendered walk
AREA = (-16.0, 20.0, -22.0, 14.0)
ELLIPSOID, CAPSULE, CYLINDER, SPHERE = (
    mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_SPHERE,
)


def add_scenery(spec: mujoco.MjSpec, seed: int = 0) -> None:
    """Add rainforest decoration to the world body of ``spec``."""
    rng = np.random.default_rng(seed)
    world = spec.worldbody

    def geom(type_, size, pos, rgb, quat=(1, 0, 0, 0), fromto=None):
        dist = np.hypot(pos[0] - 3.0, pos[1] - 1.0)
        rgb = np.clip(np.asarray(rgb) * rng.uniform(0.85, 1.15), 0, 1)
        rgb = rgb + (HAZE - rgb) * min(dist / 40.0, 0.6)  # fake aerial perspective
        g = world.add_geom(
            type=type_,
            size=list(size) + [0] * (3 - len(size)),
            pos=list(pos),
            quat=list(quat),
            rgba=[*rgb, 1],
            contype=0,
            conaffinity=0,
            group=2,
            density=0,
        )
        if fromto is not None:
            g.fromto = list(fromto)
        return g

    def spot(free: float = 0.0) -> np.ndarray:
        while True:
            x, y = rng.uniform(AREA[0], AREA[1]), rng.uniform(AREA[2], AREA[3])
            if not (
                CLEARING[0] - free < x < CLEARING[1] + free
                and CLEARING[2] - free < y < CLEARING[3] + free
            ):
                return np.array([x, y])

    def yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, [0.0, pitch, yaw], "zyx")
        return q

    for _ in range(55):  # trees with buttress roots and a high canopy
        x, y = spot(free=3.5)  # canopies must not shade the clearing
        r, h = rng.uniform(0.12, 0.35), rng.uniform(5.0, 11.0)
        bark = [0.36, 0.31, 0.25] if rng.random() < 0.6 else [0.48, 0.44, 0.38]
        geom(CYLINDER, [r, h / 2], [x, y, h / 2], bark)
        for k in range(rng.integers(3, 6)):
            a = rng.uniform(0, 2 * np.pi)
            geom(
                ELLIPSOID,
                [r * 2.2, r * 0.35, r * 2.6],
                [x + np.cos(a) * r, y + np.sin(a) * r, 0.0],
                bark,
                quat=yaw_pitch(a, 0.0),
            )
        for _ in range(rng.integers(3, 6)):
            off = rng.normal(0, 1.0, 3) * [1, 1, 0.4]
            s = rng.uniform(1.4, 2.8)
            geom(
                ELLIPSOID, [s, s, s * 0.6], [x + off[0], y + off[1], h + off[2]], [0.12, 0.30, 0.10]
            )
        if rng.random() < 0.5:  # a hanging liana
            lx = x + rng.uniform(-1, 1)
            geom(
                CAPSULE,
                [0.02],
                [0, 0, 0],
                [0.30, 0.34, 0.16],
                fromto=[
                    lx,
                    y + 0.3,
                    h,
                    lx + rng.uniform(-0.5, 0.5),
                    y + 0.4,
                    rng.uniform(0.5, 2.0),
                ],
            )

    for _ in range(110):  # bushes
        x, y = spot(free=0.3)
        for _ in range(rng.integers(3, 6)):
            s = rng.uniform(0.25, 0.6)
            off = rng.normal(0, 0.3, 2)
            green = [0.10, 0.27, 0.08] if rng.random() < 0.6 else [0.18, 0.34, 0.10]
            geom(ELLIPSOID, [s, s, s * 0.75], [x + off[0], y + off[1], s * 0.4], green)

    for _ in range(90):  # ferns: fronds fanning out and up
        x, y = spot(free=0.1)
        n, length = rng.integers(6, 10), rng.uniform(0.25, 0.5)
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 1):
            pitch = -rng.uniform(0.35, 0.8)
            c = length * np.cos(-pitch)
            geom(
                ELLIPSOID,
                [length, length * 0.18, 0.006],
                [x + np.cos(a) * c, y + np.sin(a) * c, length * np.sin(-pitch) + 0.02],
                [0.20, 0.42, 0.12],
                quat=yaw_pitch(a, pitch),
            )

    for _ in range(40):  # mossy rocks
        x, y = spot(free=0.2)
        s = rng.uniform(0.12, 0.5)
        q = rng.normal(size=4)
        geom(
            ELLIPSOID,
            [s, s * rng.uniform(0.6, 1), s * rng.uniform(0.4, 0.7)],
            [x, y, 0.0],
            [0.38, 0.40, 0.33] if rng.random() < 0.6 else [0.30, 0.38, 0.20],
            quat=q / np.linalg.norm(q),
        )

    for _ in range(10):  # fallen logs
        x, y = spot(free=1.0)
        a, length, r = rng.uniform(0, np.pi), rng.uniform(1.5, 4.0), rng.uniform(0.12, 0.25)
        d = np.array([np.cos(a), np.sin(a)]) * length / 2
        geom(
            CAPSULE,
            [r],
            [0, 0, 0],
            [0.34, 0.27, 0.18],
            fromto=[x - d[0], y - d[1], r * 0.8, x + d[0], y + d[1], r * 0.8],
        )
        geom(
            CAPSULE,
            [r * 0.9],
            [0, 0, 0],
            [0.26, 0.40, 0.14],
            fromto=[
                x - d[0] * 0.6,
                y - d[1] * 0.6,
                r * 1.15,
                x + d[0] * 0.3,
                y + d[1] * 0.3,
                r * 1.15,
            ],
        )

    for _ in range(220):  # grass tufts and flowers, also inside the clearing
        x, y = rng.uniform(-8, 12), rng.uniform(-14, 8)
        if rng.random() < 0.75:
            for a in rng.uniform(0, 2 * np.pi, 3):
                h = rng.uniform(0.06, 0.14)
                geom(
                    ELLIPSOID,
                    [0.008, 0.02, h],
                    [x + 0.02 * np.cos(a), y + 0.02 * np.sin(a), h * 0.6],
                    [0.30, 0.50, 0.16],
                    quat=yaw_pitch(a, rng.uniform(-0.3, 0.3)),
                )
        else:
            h = rng.uniform(0.08, 0.18)
            geom(CAPSULE, [0.004], [0, 0, 0], [0.25, 0.45, 0.15], fromto=[x, y, 0, x, y, h])
            petal = [[0.90, 0.25, 0.30], [0.98, 0.70, 0.20], [0.95, 0.90, 0.85], [0.70, 0.35, 0.80]]
            geom(SPHERE, [0.022], [x, y, h], petal[rng.integers(4)])
