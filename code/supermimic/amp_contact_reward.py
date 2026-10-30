"""Differentiable flat-ground costs from the actual G1 foot collision geometry.

These are geometric/kinematic proxies for soft landing and penetration, not
predictions of contact force. Geometry is read once from a compiled MuJoCo model.
A sphere uses two identical centers; a capsule uses its two endpoint sphere
centers. Its exact minimum distance to a horizontal plane is the minimum endpoint
center height minus its radius. Kinematic velocities are at body frame origins.
"""

from __future__ import annotations

import math
import re

import mujoco
import numpy as np
import torch


_FOOT_NAMES = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))


def _basename(name):
    return str(name).split("/")[-1]


def _finite_parameter(name, value, *, positive=False):
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {relation}")


class FootContactCosts:
    """Fourteen sphere/capsule foot geoms in left-then-right, 1..7 order.

    Use only when an associated reward weight is enabled. No MuJoCo operations
    occur in samples()/costs(), so real and imagined states share the same cost.
    """

    @classmethod
    def from_model(cls, model, body_names, device="cpu"):
        names = tuple(_basename(name) for name in body_names)
        if len(set(names)) != len(names):
            raise ValueError("Kinematic body names must be unique after removing prefixes")
        found = {}
        for geom in range(model.ngeom):
            name = _basename(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
            if re.fullmatch(r"(?:left|right)_foot[1-7]_collision", name):
                if name in found:
                    raise ValueError(f"Duplicate foot collision geom: {name}")
                found[name] = geom
        if set(found) != set(_FOOT_NAMES):
            missing = sorted(set(_FOOT_NAMES) - set(found))
            raise ValueError(f"Expected all 14 G1 foot collision geoms; missing {missing}")

        ids, indices, centers, radii, types = [], [], [], [], []
        for name in _FOOT_NAMES:
            geom = found[name]
            kind = int(model.geom_type[geom])
            if kind not in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                raise ValueError(f"Foot geom {name} must be a sphere or capsule")
            body = _basename(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or "")
            if body not in names:
                raise ValueError(f"Foot geom {name} body {body!r} is absent from kinematic body_names")
            radius = float(model.geom_size[geom, 0])
            _finite_parameter(f"{name} radius", radius, positive=True)
            offset = np.asarray(model.geom_pos[geom], dtype=np.float64).copy()
            quat = np.asarray(model.geom_quat[geom], dtype=np.float64)
            if not np.isfinite(offset).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
                raise ValueError(f"Non-finite or invalid geometry transform: {name}")
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                raise ValueError(f"Foot geom {name} has collisions disabled")
            half = float(model.geom_size[geom, 1]) if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.
            _finite_parameter(f"{name} capsule half-length", half)
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, quat / np.linalg.norm(quat))
            axis = rotation.reshape(3, 3)[:, 2]
            centers.append(np.stack((offset - half * axis, offset + half * axis)))
            radii.append(radius)
            indices.append(names.index(body))
            ids.append(geom)
            types.append("capsule" if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else "sphere")

        # Match the actual fixed plane, including a translated/rotated fixed body.
        ground = np.flatnonzero((model.body_weldid[model.geom_bodyid] == 0)
                               & (model.geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE)))
        if not len(ground):
            raise ValueError("Contact costs require a fixed horizontal plane")
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        heights = []
        for geom in ground:
            normal = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            if not np.allclose(normal, (0., 0., 1.), atol=1e-6):
                raise ValueError("Contact costs support only upward-facing horizontal planes")
            heights.append(float(data.geom_xpos[geom, 2]))
        if not np.isfinite(heights).all() or not np.allclose(heights, heights[0], atol=1e-7):
            raise ValueError("Fixed ground planes must share one finite height")

        obj = cls()
        obj.body_names = names
        obj.body_indices = torch.tensor(indices, dtype=torch.long, device=device)
        obj.local_centers = torch.tensor(np.stack(centers), dtype=torch.float32, device=device)
        obj.radii = torch.tensor(radii, dtype=torch.float32, device=device)
        obj.ground_height = heights[0]
        obj.metadata = {"geometry_names": list(_FOOT_NAMES), "geometry_ids": ids,
                        "geometry_types": types, "radius_m": radii,
                        "ground_height_m": obj.ground_height,
                        "aggregation": "Mean of 7 geoms per foot, then sum of the two feet",
                        "landing": "Mean of both endpoint-sphere costs per geom",
                        "penetration": "Largest endpoint-sphere depth per geom beyond slop",
                        "interpretation": "Geometric/kinematic cost, not a contact-force prediction"}
        return obj

    def samples(self, kin):
        """Return centers/velocities (...,14,2,3), clearance (...,14,2),
        and minimum_clearance (...,2), where the last axis of the latter is feet.
        The two endpoint spheres are identical for sphere geoms.
        """
        positions, rotations, velocities, angular = kin
        if (positions.shape[-2:] != (len(self.body_names), 3)
                or rotations.shape != positions.shape[:-1] + (3, 3)
                or velocities.shape != positions.shape or angular.shape != positions.shape):
            raise ValueError("Expected matching body-origin kinematics (..., bodies, 3)")
        ids = self.body_indices.to(device=positions.device)
        local = self.local_centers.to(device=positions.device, dtype=positions.dtype)
        radii = self.radii.to(device=positions.device, dtype=positions.dtype)
        rotation = rotations.index_select(-3, ids)
        offsets = (rotation.unsqueeze(-3) @ local.unsqueeze(-1)).squeeze(-1)
        centers = positions.index_select(-2, ids).unsqueeze(-2) + offsets
        omega = angular.index_select(-2, ids).unsqueeze(-2).expand_as(offsets)
        point_velocity = velocities.index_select(-2, ids).unsqueeze(-2) + torch.cross(omega, offsets, dim=-1)
        clearance = centers[..., 2] - radii[:, None] - self.ground_height
        minimum = clearance.reshape(clearance.shape[:-2] + (2, 7, 2)).amin(dim=(-1, -2))
        return {"centers": centers, "velocities": point_velocity,
                "clearance": clearance, "minimum_clearance": minimum}

    def costs(self, kin, *, landing_height=0.02, landing_width=0.01,
              penetration_scale=0.01, penetration_slop=0.0005):
        """Positive per-state costs, averaged over geoms per foot, then summed.

        landing has units (m/s)^2; penetration is dimensionless. The landing gate
        remains differentiable, including its height derivative. Weights and dt
        belong to the caller. At upward/zero velocity landing cost is exactly zero.
        """
        _finite_parameter("landing_height", landing_height)
        _finite_parameter("landing_width", landing_width, positive=True)
        _finite_parameter("penetration_scale", penetration_scale, positive=True)
        _finite_parameter("penetration_slop", penetration_slop)
        sample = self.samples(kin)
        clearance = sample["clearance"]
        gate = torch.sigmoid((landing_height - clearance) / landing_width)
        downward = (-sample["velocities"][..., 2]).clamp_min(0)
        landing = (gate * downward.square()).mean(-1)
        depth = (-clearance - penetration_slop).clamp_min(0).amax(-1)
        penetration = (depth / penetration_scale).square()
        def aggregate(value):
            return value.reshape(value.shape[:-1] + (2, 7)).mean(-1).sum(-1)
        return {"landing": aggregate(landing), "penetration": aggregate(penetration)}
