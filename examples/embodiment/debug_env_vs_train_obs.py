#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import Any


def _bootstrap_paths() -> tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    embodied_path = script_path.parent
    rlinf_root = embodied_path.parents[1]
    repo_root = rlinf_root.parent

    candidates = [
        rlinf_root,
        repo_root / "openpi-RLT" / "src",
        repo_root.parent / "openpi-RLT" / "src",
    ]
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    os.environ.setdefault("EMBODIED_PATH", str(embodied_path))
    return repo_root, rlinf_root, embodied_path


REPO_ROOT, RLINF_ROOT, EMBODIED_PATH = _bootstrap_paths()

import numpy as np
import torch
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


_INDEX_OVERRIDE_RE = re.compile(
    r"^(?P<prefix>[\w.]+)\[(?P<index>\d+)\]\.(?P<suffix>[^=]+)=(?P<value>.*)$"
)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def _to_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (bool, np.bool_)):
        return torch.tensor(bool(value))
    if isinstance(value, np.generic):
        return torch.as_tensor(value.item())
    try:
        return torch.as_tensor(value)
    except Exception:
        return None


def _stats(name: str, value: Any) -> None:
    tensor = _to_tensor(value)
    if tensor is None:
        if isinstance(value, (str, bytes)):
            print(f"{name}: {value!r}")
        else:
            print(f"{name}: <non-numeric {type(value).__name__}>")
        return

    dtype = tensor.dtype
    device = tensor.device
    tensor = tensor.detach().cpu().to(torch.float32)
    shape = tuple(tensor.shape)
    flat = tensor.reshape(-1)
    finite_vals = flat[torch.isfinite(flat)]
    if finite_vals.numel() == 0:
        print(f"{name}: shape={shape} dtype={dtype} device={device} no_finite_values")
        return

    std = finite_vals.std(unbiased=False).item() if finite_vals.numel() > 1 else 0.0
    print(
        f"{name}: shape={shape} dtype={dtype} device={device} "
        f"mean={finite_vals.mean().item():.6f} std={std:.6f} "
        f"min={finite_vals.min().item():.6f} max={finite_vals.max().item():.6f} "
        f"abs_mean={finite_vals.abs().mean().item():.6f}"
    )


def _summarize_tree(value: Any, prefix: str, max_keys: int = 60) -> None:
    if value is None:
        print(f"{prefix}: <none>")
        return
    if isinstance(value, dict):
        print(f"{prefix}: dict(keys={list(value.keys())})")
        for idx, (key, sub_value) in enumerate(value.items()):
            if idx >= max_keys:
                print(f"{prefix}: ... skipped remaining keys")
                break
            _summarize_tree(sub_value, f"{prefix}[{key}]", max_keys=max_keys)
        return
    if isinstance(value, (str, bytes)):
        print(f"{prefix}: {value!r}")
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            print(f"{prefix}: {type(value).__name__}[0]")
            return
        if all(isinstance(item, (str, bytes)) for item in value):
            print(f"{prefix}: {type(value).__name__}[{len(value)}] first={value[0]!r}")
            return
        tensor = _to_tensor(value)
        if tensor is not None:
            _stats(prefix, value)
            return
        print(f"{prefix}: {type(value).__name__}[{len(value)}]")
        return
    _stats(prefix, value)


def _first_values(value: Any, n: int = 12) -> list[float] | None:
    tensor = _to_tensor(value)
    if tensor is None:
        return None
    flat = tensor.detach().cpu().to(torch.float32).reshape(-1)
    return [round(float(item), 6) for item in flat[:n].tolist()]


def _select_batch(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _select_batch(sub_value, index) for key, sub_value in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, list):
        if len(value) > index:
            return value[index]
        return value
    if isinstance(value, tuple):
        if len(value) > index:
            return value[index]
        return value
    tensor = _to_tensor(value)
    if tensor is not None and tensor.ndim > 0 and tensor.shape[0] > index:
        return value[index]
    return value


def _nested_get(value: Any, path: tuple[str, ...]) -> Any:
    curr = value
    for key in path:
        if not isinstance(curr, dict) or key not in curr:
            return None
        curr = curr[key]
    return curr


def _nested_get_attr(value: Any, path: tuple[str, ...]) -> Any:
    curr = value
    for key in path:
        if curr is None:
            return None
        try:
            curr = getattr(curr, key)
        except Exception as exc:
            return f"<error reading {'.'.join(path)}: {exc}>"
    return curr


def _dict_keys(value: Any) -> list[str] | None:
    if isinstance(value, dict):
        return list(value.keys())
    return None


def _extract_raw_env_debug_fields(raw_obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_obs_keys": _dict_keys(raw_obs),
        "extra_keys": _dict_keys(raw_obs.get("extra")),
        "agent_keys": _dict_keys(raw_obs.get("agent")),
        "extra.tcp_pose": _nested_get(raw_obs, ("extra", "tcp_pose")),
        "agent.tcp_pose": _nested_get(raw_obs, ("agent", "tcp_pose")),
        "agent.qpos": _nested_get(raw_obs, ("agent", "qpos")),
    }


def _extract_live_env_debug_fields(env: Any) -> dict[str, Any]:
    raw_env = getattr(env, "env", None)
    unwrapped = getattr(raw_env, "unwrapped", raw_env)
    return {
        "live_env_type": type(unwrapped).__name__ if unwrapped is not None else None,
        "agent_type": type(getattr(unwrapped, "agent", None)).__name__
        if unwrapped is not None
        else None,
        "agent.tcp.pose.raw_pose": _nested_get_attr(
            unwrapped,
            ("agent", "tcp", "pose", "raw_pose"),
        ),
        "agent.tcp.pose.p": _nested_get_attr(unwrapped, ("agent", "tcp", "pose", "p")),
        "agent.tcp.pose.q": _nested_get_attr(unwrapped, ("agent", "tcp", "pose", "q")),
        "agent.robot.pose.raw_pose": _nested_get_attr(
            unwrapped,
            ("agent", "robot", "pose", "raw_pose"),
        ),
        "agent.ee_pose_at_robot_base.raw_pose": _nested_get_attr(
            unwrapped,
            ("agent", "ee_pose_at_robot_base", "raw_pose"),
        ),
    }


def _as_float_tensor(value: Any) -> torch.Tensor | None:
    tensor = _to_tensor(value)
    if tensor is None:
        return None
    return tensor.detach().cpu().to(torch.float32)


def _compare_numeric(name: str, train_value: Any, env_value: Any) -> None:
    train_tensor = _as_float_tensor(train_value)
    env_tensor = _as_float_tensor(env_value)
    if train_tensor is None or env_tensor is None:
        print(f"{name}: cannot_compare_non_numeric")
        return

    print(
        f"{name}: train_shape={tuple(train_tensor.shape)} "
        f"env_shape={tuple(env_tensor.shape)}"
    )
    print(f"{name}: train_first={_first_values(train_tensor)}")
    print(f"{name}: env_first={_first_values(env_tensor)}")

    if train_tensor.shape != env_tensor.shape:
        print(f"{name}: shape_mismatch_skip_diff")
        return

    diff = (train_tensor - env_tensor).abs()
    _stats(f"{name}.abs_diff", diff)


def _image_layout(name: str, value: Any) -> None:
    tensor = _to_tensor(value)
    if tensor is None:
        print(f"{name}: cannot_parse_image")
        return
    shape = tuple(tensor.shape)
    layout = "unknown"
    if len(shape) >= 3:
        if shape[-1] in (1, 3, 4):
            layout = "HWC-like"
        elif shape[-3] in (1, 3, 4):
            layout = "CHW-like"
    _stats(name, tensor)
    print(f"{name}.layout_guess={layout}")


def _prompt_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _prompt_to_string(value[0])
    if hasattr(value, "item"):
        try:
            return str(value.item())
        except Exception:
            pass
    return str(value)


def _quat_xyzw_to_rpy_np(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _quat_wxyz_to_rotvec_np(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)

    quat = quat / quat_norm
    if quat[0] < 0:
        quat = -quat

    w = np.clip(quat[0], -1.0, 1.0)
    xyz = quat[1:]
    sin_half_angle = np.linalg.norm(xyz)
    if sin_half_angle < 1e-8:
        return (2.0 * xyz).astype(np.float32)

    angle = 2.0 * np.arctan2(sin_half_angle, w)
    return (xyz / sin_half_angle * angle).astype(np.float32)


def _rotvec_to_quat_wxyz_np(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotvec / angle
    half_angle = 0.5 * angle
    return np.concatenate(
        [[np.cos(half_angle)], axis * np.sin(half_angle)],
        axis=0,
    ).astype(np.float64)


def _rotvec_delta_np(current_rotvec: np.ndarray, target_rotvec: np.ndarray) -> np.ndarray:
    current_quat = _rotvec_to_quat_wxyz_np(current_rotvec)
    target_quat = _rotvec_to_quat_wxyz_np(target_rotvec)
    delta_quat = _quat_wxyz_multiply(target_quat, _quat_wxyz_inverse(current_quat))
    return _quat_wxyz_to_rotvec_np(delta_quat)


def _euler_xyz_to_rotvec_np(delta_rpy: np.ndarray) -> np.ndarray:
    delta_rpy = np.asarray(delta_rpy, dtype=np.float64)
    roll = delta_rpy[:, 0]
    pitch = delta_rpy[:, 1]
    yaw = delta_rpy[:, 2]

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    xyz = np.stack([qx, qy, qz], axis=-1)
    norm_xyz = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm_xyz, np.abs(qw)[..., None])
    safe_axis = np.divide(
        xyz,
        np.clip(norm_xyz, 1e-8, None),
        out=np.zeros_like(xyz),
        where=norm_xyz > 1e-8,
    )
    return (safe_axis * angle).astype(np.float32)


def _quat_wxyz_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_wxyz_inverse(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    denom = np.dot(quat, quat)
    if denom < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64) / denom


def _quat_wxyz_rotate(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vec_quat = np.concatenate([[0.0], np.asarray(vector, dtype=np.float64)])
    rotated = _quat_wxyz_multiply(
        _quat_wxyz_multiply(quat_wxyz, vec_quat),
        _quat_wxyz_inverse(quat_wxyz),
    )
    return rotated[1:]


def _pose_at_base_frame(tcp_pose: Any, robot_pose: Any) -> np.ndarray | None:
    tcp_tensor = _as_float_tensor(tcp_pose)
    robot_tensor = _as_float_tensor(robot_pose)
    if tcp_tensor is None or robot_tensor is None:
        return None

    tcp = tcp_tensor.reshape(-1).numpy()
    robot = robot_tensor.reshape(-1).numpy()
    if tcp.shape[0] < 7 or robot.shape[0] < 7:
        return None

    robot_pos = robot[:3].astype(np.float64)
    robot_quat = robot[3:7].astype(np.float64)
    tcp_pos = tcp[:3].astype(np.float64)
    tcp_quat = tcp[3:7].astype(np.float64)
    robot_quat_inv = _quat_wxyz_inverse(robot_quat)

    base_pos = _quat_wxyz_rotate(robot_quat_inv, tcp_pos - robot_pos)
    base_quat = _quat_wxyz_multiply(robot_quat_inv, tcp_quat)
    base_quat = base_quat / max(np.linalg.norm(base_quat), 1e-8)
    return np.concatenate([base_pos, base_quat], axis=0).astype(np.float32)


def _candidate_state_from_pose(
    pose: Any,
    qpos: Any,
    quat_order: str,
    rotation_repr: str = "euler",
) -> np.ndarray | None:
    pose_tensor = _as_float_tensor(pose)
    qpos_tensor = _as_float_tensor(qpos)
    if pose_tensor is None or qpos_tensor is None:
        return None

    pose_np = pose_tensor.reshape(-1).numpy()
    qpos_np = qpos_tensor.reshape(-1).numpy()
    if pose_np.shape[0] < 7 or qpos_np.shape[0] < 9:
        return None

    quat = pose_np[3:7]
    if quat_order == "xyzw":
        quat_xyzw = quat
        quat_wxyz = quat[[3, 0, 1, 2]]
    elif quat_order == "wxyz":
        quat_xyzw = quat[[1, 2, 3, 0]]
        quat_wxyz = quat
    else:
        raise ValueError(f"Unsupported quat_order={quat_order}")

    if rotation_repr == "euler":
        rot = _quat_xyzw_to_rpy_np(quat_xyzw)
    elif rotation_repr == "rotvec":
        rot = _quat_wxyz_to_rotvec_np(quat_wxyz)
    else:
        raise ValueError(f"Unsupported rotation_repr={rotation_repr}")

    fingers = qpos_np[7:9].astype(np.float32)
    return np.concatenate([pose_np[:3].astype(np.float32), rot, fingers], axis=0)


def _split_index_overrides(overrides: list[str]) -> tuple[list[str], list[tuple[str, int, str, str]]]:
    hydra_overrides: list[str] = []
    index_overrides: list[tuple[str, int, str, str]] = []
    for override in overrides:
        match = _INDEX_OVERRIDE_RE.match(override)
        if match is None:
            hydra_overrides.append(override)
            continue
        index_overrides.append(
            (
                match.group("prefix"),
                int(match.group("index")),
                match.group("suffix"),
                match.group("value"),
            )
        )
    return hydra_overrides, index_overrides


def _update_cfg(cfg: Any, key: str, value: Any) -> None:
    with open_dict(cfg):
        OmegaConf.update(cfg, key, value, merge=True, force_add=True)


def _set_openpi_data_field(cfg: Any, key: str, value: Any) -> None:
    updated = False
    for prefix in ("actor.model.openpi_data", "actor.openpi_data"):
        if OmegaConf.select(cfg, prefix) is not None:
            _update_cfg(cfg, f"{prefix}.{key}", value)
            updated = True
    if not updated:
        _update_cfg(cfg, f"actor.model.openpi_data.{key}", value)


def _sync_actor_openpi_data_to_model(cfg: Any) -> None:
    model_data = OmegaConf.select(cfg, "actor.model.openpi_data")
    actor_data = OmegaConf.select(cfg, "actor.openpi_data")
    if model_data is None and actor_data is not None:
        _update_cfg(
            cfg,
            "actor.model.openpi_data",
            OmegaConf.to_container(actor_data, resolve=True),
        )


def _apply_index_overrides(cfg: Any, index_overrides: list[tuple[str, int, str, str]]) -> None:
    for prefix, index, suffix, value in index_overrides:
        if prefix == "data.train_data_paths" and index == 0 and suffix == "dataset_path":
            _set_openpi_data_field(cfg, "repo_id", value)
            continue

        container = OmegaConf.select(cfg, prefix)
        if container is None:
            raise KeyError(f"Override target does not exist: {prefix}[{index}].{suffix}")
        if index >= len(container):
            raise IndexError(
                f"Override target index out of range: {prefix}[{index}].{suffix}"
            )
        OmegaConf.update(container[index], suffix, value, merge=True)


def _apply_direct_args(cfg: Any, args: argparse.Namespace) -> None:
    if args.model_path:
        _update_cfg(cfg, "actor.model.model_path", args.model_path)
        if OmegaConf.select(cfg, "rollout.model.model_path") is not None:
            _update_cfg(cfg, "rollout.model.model_path", args.model_path)

    repo_id = args.repo_id or args.dataset_path
    if repo_id:
        _set_openpi_data_field(cfg, "repo_id", repo_id)
    if args.norm_stats_path:
        _set_openpi_data_field(cfg, "norm_stats_path", args.norm_stats_path)

    _sync_actor_openpi_data_to_model(cfg)

    env_cfg = OmegaConf.select(cfg, f"env.{args.env_split}")
    if env_cfg is None:
        raise KeyError(f"env.{args.env_split} does not exist in config.")

    with open_dict(env_cfg):
        env_cfg.total_num_envs = args.num_envs
        env_cfg.video_cfg.save_video = False
        env_cfg.auto_reset = False

    if not args.keep_compile:
        _update_cfg(cfg, "actor.model.openpi.pytorch_compile_mode", None)


def _compose_cfg(args: argparse.Namespace):
    config_dir = RLINF_ROOT / "examples" / "embodiment" / "config"
    hydra_overrides, index_overrides = _split_index_overrides(list(args.overrides))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name=args.config_name, overrides=hydra_overrides)
    _apply_index_overrides(cfg, index_overrides)
    _apply_direct_args(cfg, args)
    OmegaConf.resolve(cfg)
    return cfg


def _openpi_data_kwargs(cfg: Any) -> Any:
    data_kwargs = OmegaConf.select(cfg, "actor.model.openpi_data")
    if data_kwargs is not None:
        return data_kwargs
    data_kwargs = OmegaConf.select(cfg, "actor.openpi_data")
    if data_kwargs is not None:
        return data_kwargs
    return None


def _maybe_set_lerobot_home(cfg: Any) -> None:
    if os.environ.get("HF_LEROBOT_HOME"):
        return
    data_kwargs = _openpi_data_kwargs(cfg)
    repo_id = None
    if data_kwargs is not None:
        repo_id = data_kwargs.get("repo_id")
    if repo_id and os.path.isabs(str(repo_id)):
        os.environ["HF_LEROBOT_HOME"] = os.path.dirname(str(repo_id))


def _load_train_samples(
    cfg: Any,
    sample_index: int,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    import openpi.training.data_loader as openpi_data_loader

    config = get_openpi_config(
        cfg.actor.model.openpi.config_name,
        model_path=cfg.actor.model.model_path,
        batch_size=1,
        data_kwargs=_openpi_data_kwargs(cfg),
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    source_dataset = openpi_data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    transformed_dataset = openpi_data_loader.transform_dataset(
        source_dataset,
        data_config,
        skip_norm_stats=False,
    )
    return (
        source_dataset[sample_index],
        transformed_dataset[sample_index],
        data_config,
        source_dataset,
    )


def _build_env(cfg: Any, args: argparse.Namespace):
    env_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.env[args.env_split], resolve=True)
    )
    with open_dict(env_cfg):
        env_cfg.total_num_envs = args.num_envs
        env_cfg.video_cfg.save_video = False
        env_cfg.auto_reset = False
    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    return env_cls(
        cfg=env_cfg,
        num_envs=args.num_envs,
        seed_offset=args.seed_offset,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )


def _reset_env_with_raw(
    env: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    options = None
    if args.episode_id is not None:
        options = {
            "episode_id": torch.full(
                (args.num_envs,),
                int(args.episode_id),
                dtype=torch.long,
                device=env.device,
            )
        }

    if options is None:
        reset_options = (
            {"episode_id": env.reset_state_ids}
            if getattr(env, "use_fixed_reset_state_ids", False)
            else {}
        )
        raw_obs, infos = env.env.reset(seed=args.seed, options=reset_options)
    else:
        raw_obs, infos = env.env.reset(seed=args.seed, options=options)

    env._show_goal_site_visual()
    env_obs = env._wrap_obs(raw_obs, infos=infos)
    env._reset_peg_insertion_event_state()
    env._reset_metrics()
    raw_debug = _extract_raw_env_debug_fields(raw_obs)
    return env_obs, infos, raw_obs, raw_debug


def _make_reset_options(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    if args.episode_id is not None:
        return {
            "episode_id": torch.full(
                (args.num_envs,),
                int(args.episode_id),
                dtype=torch.long,
                device=env.device,
            )
        }
    if getattr(env, "use_fixed_reset_state_ids", False):
        return {"episode_id": env.reset_state_ids}
    return {}


def _reset_env(env: Any, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_obs, infos = env.env.reset(seed=args.seed, options=_make_reset_options(env, args))
    env._show_goal_site_visual()
    env_obs = env._wrap_obs(raw_obs, infos=infos)
    env._reset_peg_insertion_event_state()
    env._reset_metrics()
    return env_obs, infos


def _transform_env_obs(model: Any, env_obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    to_process_obs = model.obs_processor(env_obs)
    transformed_obs = model.input_transform(to_process_obs, transpose=False)
    return to_process_obs, transformed_obs


def _compare_prompts(source_sample: dict[str, Any], env_obs: dict[str, Any], env_index: int) -> None:
    train_prompt = source_sample.get("prompt", source_sample.get("task", ""))
    env_prompt = env_obs.get("task_descriptions", "")
    env_prompt = _select_batch(env_prompt, env_index)
    train_prompt_str = _prompt_to_string(train_prompt)
    env_prompt_str = _prompt_to_string(env_prompt)
    print(f"train_prompt={train_prompt_str!r}")
    print(f"env_prompt={env_prompt_str!r}")
    print(f"prompt_equal={train_prompt_str == env_prompt_str}")


def _compare_core_fields(
    source_sample: dict[str, Any],
    train_sample: dict[str, Any],
    env_obs: dict[str, Any],
    env_transformed_sample: dict[str, Any],
    env_index: int,
) -> None:
    _print_section("Prompt Compare")
    _compare_prompts(source_sample, env_obs, env_index)

    _print_section("Raw State Compare")
    _compare_numeric(
        "raw_state",
        source_sample.get("state"),
        _select_batch(env_obs.get("states"), env_index),
    )

    _print_section("Transformed State Compare")
    _compare_numeric(
        "transformed_state",
        train_sample.get("state"),
        env_transformed_sample.get("state"),
    )

    _print_section("Tokenized Prompt Compare")
    _compare_numeric(
        "tokenized_prompt",
        train_sample.get("tokenized_prompt"),
        env_transformed_sample.get("tokenized_prompt"),
    )
    _compare_numeric(
        "tokenized_prompt_mask",
        train_sample.get("tokenized_prompt_mask"),
        env_transformed_sample.get("tokenized_prompt_mask"),
    )

    _print_section("Raw Image Layout")
    _image_layout("source.image", source_sample.get("image"))
    _image_layout("source.wrist_image", source_sample.get("wrist_image"))
    _image_layout(
        "env.main_images[selected]",
        _select_batch(env_obs.get("main_images"), env_index),
    )
    _image_layout(
        "env.wrist_images[selected]",
        _select_batch(env_obs.get("wrist_images"), env_index),
    )

    _print_section("Transformed Image Compare")
    train_images = train_sample.get("image", {})
    env_images = env_transformed_sample.get("image", {})
    if not isinstance(train_images, dict) or not isinstance(env_images, dict):
        print("transformed_image: missing image dict")
        return
    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        if key not in train_images or key not in env_images:
            print(f"image[{key}]: missing train={key in train_images} env={key in env_images}")
            continue
        _compare_numeric(f"image[{key}]", train_images[key], env_images[key])

    _print_section("Image Mask Compare")
    train_masks = train_sample.get("image_mask", {})
    env_masks = env_transformed_sample.get("image_mask", {})
    if isinstance(train_masks, dict) and isinstance(env_masks, dict):
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            if key in train_masks and key in env_masks:
                _compare_numeric(f"image_mask[{key}]", train_masks[key], env_masks[key])


def _print_raw_env_pose_debug(
    source_sample: dict[str, Any],
    raw_debug: dict[str, Any],
    live_debug: dict[str, Any] | None,
    env_index: int,
) -> None:
    _print_section("Raw Env Pose Debug")
    for key in ("raw_obs_keys", "extra_keys", "agent_keys"):
        print(f"{key}={raw_debug.get(key)}")

    extra_tcp_pose = _select_batch(raw_debug.get("extra.tcp_pose"), env_index)
    agent_tcp_pose = _select_batch(raw_debug.get("agent.tcp_pose"), env_index)
    agent_qpos = _select_batch(raw_debug.get("agent.qpos"), env_index)

    _stats("raw_env.extra.tcp_pose[selected]", extra_tcp_pose)
    print(f"raw_env.extra.tcp_pose.first={_first_values(extra_tcp_pose, n=7)}")
    _stats("raw_env.agent.tcp_pose[selected]", agent_tcp_pose)
    print(f"raw_env.agent.tcp_pose.first={_first_values(agent_tcp_pose, n=7)}")
    _stats("raw_env.agent.qpos[selected]", agent_qpos)
    print(f"raw_env.agent.qpos.first={_first_values(agent_qpos, n=12)}")

    live_tcp_pose = None
    live_ee_pose_at_robot_base = None
    live_robot_pose = None
    if live_debug is not None:
        print(f"live_env_type={live_debug.get('live_env_type')}")
        print(f"live_agent_type={live_debug.get('agent_type')}")
        live_tcp_pose = _select_batch(
            live_debug.get("agent.tcp.pose.raw_pose"),
            env_index,
        )
        live_tcp_p = _select_batch(live_debug.get("agent.tcp.pose.p"), env_index)
        live_tcp_q = _select_batch(live_debug.get("agent.tcp.pose.q"), env_index)
        live_robot_pose = _select_batch(
            live_debug.get("agent.robot.pose.raw_pose"),
            env_index,
        )
        live_ee_pose_at_robot_base = _select_batch(
            live_debug.get("agent.ee_pose_at_robot_base.raw_pose"),
            env_index,
        )
        _stats("live_env.agent.tcp.pose.raw_pose[selected]", live_tcp_pose)
        print(f"live_env.agent.tcp.pose.raw_pose.first={_first_values(live_tcp_pose, n=7)}")
        _stats("live_env.agent.tcp.pose.p[selected]", live_tcp_p)
        print(f"live_env.agent.tcp.pose.p.first={_first_values(live_tcp_p, n=3)}")
        _stats("live_env.agent.tcp.pose.q[selected]", live_tcp_q)
        print(f"live_env.agent.tcp.pose.q.first={_first_values(live_tcp_q, n=4)}")
        _stats("live_env.agent.robot.pose.raw_pose[selected]", live_robot_pose)
        print(f"live_env.agent.robot.pose.raw_pose.first={_first_values(live_robot_pose, n=7)}")
        _stats(
            "live_env.agent.ee_pose_at_robot_base.raw_pose[selected]",
            live_ee_pose_at_robot_base,
        )
        print(
            "live_env.agent.ee_pose_at_robot_base.raw_pose.first="
            f"{_first_values(live_ee_pose_at_robot_base, n=7)}"
        )

    _print_section("Candidate State From Raw Env Pose")
    candidates = {
        "extra_tcp_pose_as_xyzw": _candidate_state_from_pose(
            extra_tcp_pose,
            agent_qpos,
            quat_order="xyzw",
        ),
        "extra_tcp_pose_as_wxyz": _candidate_state_from_pose(
            extra_tcp_pose,
            agent_qpos,
            quat_order="wxyz",
        ),
        "extra_tcp_pose_wxyz_as_rotvec": _candidate_state_from_pose(
            extra_tcp_pose,
            agent_qpos,
            quat_order="wxyz",
            rotation_repr="rotvec",
        ),
        "agent_tcp_pose_as_xyzw": _candidate_state_from_pose(
            agent_tcp_pose,
            agent_qpos,
            quat_order="xyzw",
        ),
        "agent_tcp_pose_as_wxyz": _candidate_state_from_pose(
            agent_tcp_pose,
            agent_qpos,
            quat_order="wxyz",
        ),
        "live_tcp_pose_as_xyzw": _candidate_state_from_pose(
            live_tcp_pose,
            agent_qpos,
            quat_order="xyzw",
        ),
        "live_tcp_pose_as_wxyz": _candidate_state_from_pose(
            live_tcp_pose,
            agent_qpos,
            quat_order="wxyz",
        ),
        "live_tcp_pose_wxyz_as_rotvec": _candidate_state_from_pose(
            live_tcp_pose,
            agent_qpos,
            quat_order="wxyz",
            rotation_repr="rotvec",
        ),
        "live_tcp_pose_at_robot_base_wxyz_as_rotvec": _candidate_state_from_pose(
            _pose_at_base_frame(live_tcp_pose, live_robot_pose),
            agent_qpos,
            quat_order="wxyz",
            rotation_repr="rotvec",
        ),
        "live_ee_pose_at_robot_base_as_xyzw": _candidate_state_from_pose(
            live_ee_pose_at_robot_base,
            agent_qpos,
            quat_order="xyzw",
        ),
        "live_ee_pose_at_robot_base_as_wxyz": _candidate_state_from_pose(
            live_ee_pose_at_robot_base,
            agent_qpos,
            quat_order="wxyz",
        ),
        "live_ee_pose_at_robot_base_wxyz_as_rotvec": _candidate_state_from_pose(
            live_ee_pose_at_robot_base,
            agent_qpos,
            quat_order="wxyz",
            rotation_repr="rotvec",
        ),
    }

    train_state = source_sample.get("state")
    for name, candidate in candidates.items():
        if candidate is None:
            print(f"{name}: <unavailable>")
            continue
        _stats(name, candidate)
        print(f"{name}.first={_first_values(candidate)}")
        _compare_numeric(f"{name}_vs_train_raw_state", train_state, candidate)


def _action_chunk_candidates(action_chunk: Any) -> dict[str, np.ndarray]:
    actions = np.asarray(_as_float_tensor(action_chunk), dtype=np.float32).copy()
    if actions.ndim != 2 or actions.shape[-1] != 7:
        return {}

    euler_to_rotvec = actions.copy()
    euler_to_rotvec[:, 3:6] = _euler_xyz_to_rotvec_np(euler_to_rotvec[:, 3:6])

    gripper_01_to_pm1 = euler_to_rotvec.copy()
    gripper_01_to_pm1[:, -1] = np.clip(2.0 * gripper_01_to_pm1[:, -1] - 1.0, -1.0, 1.0)

    no_rot_convert = actions.copy()
    no_rot_convert[:, -1] = np.clip(no_rot_convert[:, -1], -1.0, 1.0)

    scaled_x10 = euler_to_rotvec.copy()
    scaled_x10[:, :6] *= 10.0
    scaled_x10[:, :6] = np.clip(scaled_x10[:, :6], -1.0, 1.0)

    return {
        "dataset_xyz_euler_to_rotvec_gripper_raw": euler_to_rotvec,
        "dataset_xyz_euler_to_rotvec_x10_gripper_raw": scaled_x10,
        "dataset_xyz_euler_to_rotvec_gripper_01_to_pm1": gripper_01_to_pm1,
        "dataset_raw_no_rot_convert_gripper_raw": no_rot_convert,
    }


def _format_step_state(value: Any) -> list[float] | None:
    return _first_values(value, n=8)


def _truthy_first(value: Any) -> bool:
    tensor = _to_tensor(value)
    if tensor is None:
        return False
    flat = tensor.detach().cpu().reshape(-1)
    if flat.numel() == 0:
        return False
    return bool(flat[0].item())


def _float_first(value: Any) -> float | None:
    tensor = _to_tensor(value)
    if tensor is None:
        return None
    flat = tensor.detach().cpu().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return None
    return float(flat[0].item())


def _dataset_frame_index(sample: dict[str, Any]) -> int | None:
    frame_index = _to_tensor(sample.get("frame_index"))
    if frame_index is None:
        return None
    return int(frame_index.detach().cpu().reshape(-1)[0].item())


def _debug_gt_action_replay(
    env: Any,
    args: argparse.Namespace,
    source_dataset: Any,
    sample_index: int,
    source_sample: dict[str, Any],
) -> None:
    actions = _as_float_tensor(source_sample.get("actions"))
    frame_index = _to_tensor(source_sample.get("frame_index"))
    if actions is None or actions.ndim != 2 or actions.shape[-1] != 7:
        print("gt_action_replay: source actions unavailable")
        return
    if frame_index is None:
        print("gt_action_replay: source frame_index unavailable")
        return

    start_frame = int(frame_index.detach().cpu().reshape(-1)[0].item())
    candidates = _action_chunk_candidates(actions)
    if not candidates:
        print("gt_action_replay: no action candidates")
        return

    _print_section("GT Action Replay")
    print(f"replay_start_sample_index={sample_index}")
    print(f"replay_start_frame_index={start_frame}")
    print(f"replay_action_first_step={_first_values(actions[0], n=7)}")

    target_states: dict[int, Any] = {}
    for step_idx in range(actions.shape[0] + 1):
        target_index = sample_index + step_idx
        try:
            target_sample = source_dataset[target_index]
        except Exception as exc:
            print(f"target_state[{step_idx}]: <unavailable {exc}>")
            continue

        target_frame_index = _to_tensor(target_sample.get("frame_index"))
        if target_frame_index is not None:
            target_frame = int(target_frame_index.detach().cpu().reshape(-1)[0].item())
            if target_frame != start_frame + step_idx:
                print(
                    f"target_state[{step_idx}]: skipped non-contiguous frame "
                    f"{target_frame} expected {start_frame + step_idx}"
                )
                continue
        target_states[step_idx] = target_sample.get("state")

    for name, env_actions_np in candidates.items():
        env_obs, _ = _reset_env(env, args)
        start_state = _select_batch(env_obs.get("states"), args.env_index)
        print(f"{name}.start_state={_format_step_state(start_state)}")
        if 0 in target_states:
            _compare_numeric(f"{name}.step0_state_vs_dataset", target_states[0], start_state)

        env_actions = torch.as_tensor(
            env_actions_np,
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        for step_idx in range(env_actions.shape[1]):
            env_obs, step_reward, terminations, truncations, infos = env.step(
                env_actions[:, step_idx],
                auto_reset=False,
            )
            state = _select_batch(env_obs.get("states"), args.env_index)
            print(
                f"{name}.after_step[{step_idx + 1}] "
                f"state={_format_step_state(state)} "
                f"reward={_first_values(step_reward, n=1)} "
                f"terminated={_first_values(terminations, n=1)} "
                f"truncated={_first_values(truncations, n=1)} "
                f"success={_first_values(infos.get('success'), n=1)}"
            )
            if step_idx + 1 in target_states:
                _compare_numeric(
                    f"{name}.step{step_idx + 1}_state_vs_dataset",
                    target_states[step_idx + 1],
                    state,
                )


def _debug_full_gt_action_replay(
    env: Any,
    args: argparse.Namespace,
    source_dataset: Any,
    sample_index: int,
    source_sample: dict[str, Any],
) -> None:
    _print_section("Full GT Action Replay")

    start_frame = _dataset_frame_index(source_sample)
    if start_frame is None:
        print("full_gt_replay: source frame_index unavailable")
        return

    max_steps = int(args.full_gt_replay_max_steps)
    action_index = int(args.full_gt_replay_action_index)
    action_candidate = str(args.full_gt_replay_action_candidate)
    if action_index < 0:
        raise ValueError("--full-gt-replay-action-index must be non-negative.")

    print(f"full_replay_start_sample_index={sample_index}")
    print(f"full_replay_start_frame_index={start_frame}")
    print(f"full_replay_max_steps={max_steps}")
    print(f"full_replay_action_candidate={action_candidate}")
    print(f"full_replay_action_index={action_index}")

    env_obs, _ = _reset_env(env, args)
    start_state = _select_batch(env_obs.get("states"), args.env_index)
    print(f"full_replay.start_state={_format_step_state(start_state)}")
    _compare_numeric("full_replay.step0_state_vs_dataset", source_sample.get("state"), start_state)

    first_grasp_step = None
    first_consecutive_grasp_step = None
    first_prealign_step = None
    first_partial_insert_step = None
    first_success_step = None
    last_infos: dict[str, Any] = {}
    last_reward = None
    last_terminated = None
    last_truncated = None
    steps_run = 0
    stopped_reason = "max_steps"

    for step_idx in range(max_steps):
        target_index = sample_index + step_idx
        try:
            sample = source_dataset[target_index]
        except Exception as exc:
            stopped_reason = f"dataset_unavailable:{exc}"
            break

        frame = _dataset_frame_index(sample)
        if frame is not None and frame != start_frame + step_idx:
            stopped_reason = (
                f"non_contiguous_frame:{frame}:expected:{start_frame + step_idx}"
            )
            break

        actions = _as_float_tensor(sample.get("actions"))
        candidates = _action_chunk_candidates(actions)
        env_action = candidates.get(action_candidate)
        if env_action is None:
            stopped_reason = (
                f"action_candidate_unavailable:{action_candidate}:"
                f"available:{list(candidates.keys())}"
            )
            break
        if action_index >= env_action.shape[0]:
            stopped_reason = (
                f"action_index_out_of_range:{action_index}:len:{env_action.shape[0]}"
            )
            break

        action = torch.as_tensor(
            env_action[action_index],
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        env_obs, last_reward, last_terminated, last_truncated, last_infos = env.step(
            action,
            auto_reset=False,
        )
        steps_run = step_idx + 1

        if first_grasp_step is None and _truthy_first(
            last_infos.get("is_grasped_current")
        ):
            first_grasp_step = steps_run
        if first_consecutive_grasp_step is None and _truthy_first(
            last_infos.get("consecutive_grasp_once")
        ):
            first_consecutive_grasp_step = steps_run
        if first_prealign_step is None and _truthy_first(
            last_infos.get("prealign_once")
        ):
            first_prealign_step = steps_run
        if first_partial_insert_step is None and _truthy_first(
            last_infos.get("partial_insert_once")
        ):
            first_partial_insert_step = steps_run
        if first_success_step is None and _truthy_first(last_infos.get("success")):
            first_success_step = steps_run

        if (
            steps_run == 1
            or steps_run % args.full_gt_replay_log_every == 0
            or _truthy_first(last_infos.get("success"))
        ):
            state = _select_batch(env_obs.get("states"), args.env_index)
            print(
                f"full_replay.step[{steps_run}] "
                f"state={_format_step_state(state)} "
                f"reward={_first_values(last_reward, n=1)} "
                f"terminated={_first_values(last_terminated, n=1)} "
                f"truncated={_first_values(last_truncated, n=1)} "
                f"grasp={_first_values(last_infos.get('is_grasped_current'), n=1)} "
                f"success={_first_values(last_infos.get('success'), n=1)} "
                f"peg_head_hole_x={_float_first(last_infos.get('peg_head_hole_x'))} "
                f"peg_goal_yz={_float_first(last_infos.get('peg_head_goal_yz_dist'))}"
            )
            target_after_index = sample_index + steps_run
            try:
                target_after_sample = source_dataset[target_after_index]
                target_after_frame = _dataset_frame_index(target_after_sample)
                if target_after_frame == start_frame + steps_run:
                    _compare_numeric(
                        f"full_replay.step{steps_run}_state_vs_dataset",
                        target_after_sample.get("state"),
                        state,
                    )
            except Exception as exc:
                print(f"full_replay.step{steps_run}_target_state_unavailable={exc}")

        if _truthy_first(last_terminated) or _truthy_first(last_truncated):
            stopped_reason = "terminated_or_truncated"
            break

    final_state = _select_batch(env_obs.get("states"), args.env_index)
    final_target_state = None
    final_target_index = sample_index + steps_run
    try:
        final_target_sample = source_dataset[final_target_index]
        final_target_frame = _dataset_frame_index(final_target_sample)
        if final_target_frame == start_frame + steps_run:
            final_target_state = final_target_sample.get("state")
        else:
            print(
                "full_replay.final_state_vs_dataset.skipped="
                f"non_contiguous_frame:{final_target_frame}:expected:{start_frame + steps_run}"
            )
    except Exception:
        pass

    print(f"full_replay.steps_run={steps_run}")
    print(f"full_replay.stopped_reason={stopped_reason}")
    print(f"full_replay.first_grasp_step={first_grasp_step}")
    print(f"full_replay.first_consecutive_grasp_step={first_consecutive_grasp_step}")
    print(f"full_replay.first_prealign_step={first_prealign_step}")
    print(f"full_replay.first_partial_insert_step={first_partial_insert_step}")
    print(f"full_replay.first_success_step={first_success_step}")
    print(f"full_replay.final_success={_first_values(last_infos.get('success'), n=1)}")
    print(f"full_replay.final_state={_format_step_state(final_state)}")
    if final_target_state is not None:
        _compare_numeric("full_replay.final_state_vs_dataset", final_target_state, final_state)


def _debug_dataset_action_scan(source_dataset: Any, max_samples: int) -> None:
    if max_samples <= 0:
        return

    _print_section("Dataset Action Scan")
    action_mins = []
    action_maxs = []
    gripper_mins = []
    gripper_maxs = []
    first_gripper_below_half = None
    first_gripper_below_zero = None

    for sample_index in range(max_samples):
        try:
            sample = source_dataset[sample_index]
        except Exception as exc:
            print(f"scan_stopped_at={sample_index} reason={exc}")
            break

        actions = _as_float_tensor(sample.get("actions"))
        if actions is None or actions.ndim != 2 or actions.shape[-1] < 7:
            continue

        action_mins.append(actions.min().item())
        action_maxs.append(actions.max().item())
        gripper = actions[:, -1]
        gripper_mins.append(gripper.min().item())
        gripper_maxs.append(gripper.max().item())
        if first_gripper_below_half is None and (gripper < 0.5).any():
            first_gripper_below_half = sample_index
        if first_gripper_below_zero is None and (gripper < 0.0).any():
            first_gripper_below_zero = sample_index

    if not action_mins:
        print("dataset_action_scan: no valid action samples")
        return

    print(f"scan_num_samples={len(action_mins)}")
    print(f"action_min_global={min(action_mins):.6f}")
    print(f"action_max_global={max(action_maxs):.6f}")
    print(f"gripper_min_global={min(gripper_mins):.6f}")
    print(f"gripper_max_global={max(gripper_maxs):.6f}")
    print(f"first_gripper_below_0.5_sample_index={first_gripper_below_half}")
    print(f"first_gripper_below_0.0_sample_index={first_gripper_below_zero}")
    if first_gripper_below_half is not None:
        sample = source_dataset[first_gripper_below_half]
        actions = _as_float_tensor(sample.get("actions"))
        print(
            "first_gripper_below_0.5_actions_gripper="
            f"{_first_values(actions[:, -1], n=actions.shape[0])}"
        )


def _print_action_chunk_summary(name: str, actions: Any) -> None:
    tensor = _as_float_tensor(actions)
    if tensor is None:
        print(f"{name}: <unavailable>")
        return
    _stats(name, tensor)
    print(f"{name}.first_step={_first_values(tensor[0], n=min(7, tensor.shape[-1]))}")
    if tensor.ndim == 2 and tensor.shape[-1] >= 7:
        print(f"{name}.gripper={_first_values(tensor[:, -1], n=tensor.shape[0])}")
        print(f"{name}.xyz_abs_mean={tensor[:, :3].abs().mean().item():.6f}")
        print(f"{name}.rot_abs_mean={tensor[:, 3:6].abs().mean().item():.6f}")


def _debug_policy_action_on_env_obs(
    model: Any,
    env_obs: dict[str, Any],
    source_sample: dict[str, Any],
    cfg: Any,
    args: argparse.Namespace,
) -> None:
    _print_section("Policy Action On Env Observation")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with torch.no_grad():
        raw_actions, result = model.predict_action_batch(
            env_obs,
            mode="eval",
            compute_values=False,
        )

    _stats("policy.raw_actions", raw_actions)
    selected_raw = _select_batch(raw_actions, args.env_index)
    _print_action_chunk_summary("policy.raw_actions[selected]", selected_raw)

    gt_raw = source_sample.get("actions")
    _print_action_chunk_summary("dataset.raw_actions", gt_raw)
    _compare_numeric("policy_raw_vs_dataset_raw", selected_raw, gt_raw)

    env_cfg = cfg.env[args.env_split]
    prepared_actions = prepare_actions(
        raw_chunk_actions=raw_actions,
        env_type=env_cfg.env_type,
        model_type=cfg.actor.model.model_type,
        num_action_chunks=cfg.actor.model.num_action_chunks,
        action_dim=cfg.actor.model.action_dim,
        policy=cfg.actor.model.get("policy_setup", None),
        wm_env_type=env_cfg.get("wm_env_type", None),
    )
    selected_prepared = _select_batch(prepared_actions, args.env_index)
    _print_action_chunk_summary(
        "policy.prepared_env_actions[selected]",
        selected_prepared,
    )

    gt_candidates = _action_chunk_candidates(gt_raw)
    gt_prepared = gt_candidates.get("dataset_xyz_euler_to_rotvec_gripper_raw")
    if gt_prepared is not None:
        _print_action_chunk_summary("dataset.prepared_env_actions", gt_prepared)
        _compare_numeric(
            "policy_prepared_vs_dataset_prepared",
            selected_prepared,
            gt_prepared,
        )

    forward_inputs = result.get("forward_inputs", {}) if isinstance(result, dict) else {}
    if forward_inputs:
        _print_section("Policy Forward Inputs Action Debug")
        for key in ("action", "model_action"):
            value = forward_inputs.get(key)
            if value is None:
                continue
            _stats(f"forward_inputs[{key}]", value)
            selected = _select_batch(value, args.env_index)
            print(f"forward_inputs[{key}].first={_first_values(selected, n=14)}")


def _save_image(path: Path, value: Any) -> None:
    from PIL import Image

    array = np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.5:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        Image.fromarray(array).save(path)
    elif array.ndim == 3:
        Image.fromarray(array[..., :3]).save(path)


def _maybe_save_images(
    output_dir: str | None,
    source_sample: dict[str, Any],
    train_sample: dict[str, Any],
    env_obs: dict[str, Any],
    env_transformed_sample: dict[str, Any],
    env_index: int,
) -> None:
    if not output_dir:
        return

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    images_to_save = {
        "train_raw_image.png": source_sample.get("image"),
        "train_raw_wrist_image.png": source_sample.get("wrist_image"),
        "env_raw_main_image.png": _select_batch(env_obs.get("main_images"), env_index),
        "env_raw_wrist_image.png": _select_batch(env_obs.get("wrist_images"), env_index),
        "train_transformed_base_0_rgb.png": train_sample.get("image", {}).get("base_0_rgb"),
        "train_transformed_left_wrist_0_rgb.png": train_sample.get("image", {}).get("left_wrist_0_rgb"),
        "env_transformed_base_0_rgb.png": env_transformed_sample.get("image", {}).get("base_0_rgb"),
        "env_transformed_left_wrist_0_rgb.png": env_transformed_sample.get("image", {}).get("left_wrist_0_rgb"),
    }
    for filename, value in images_to_save.items():
        if value is None:
            continue
        try:
            _save_image(path / filename, value)
        except Exception as exc:
            print(f"save_image_failed {filename}: {exc}")
    print(f"saved_images_dir={path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one transformed SFT training observation against one ManiSkill "
            "environment observation after the exact OpenPI eval input transform."
        )
    )
    parser.add_argument(
        "--config-name",
        default="rlt_maniskill_pi05_sft_eval",
        help="Hydra config name under examples/embodiment/config.",
    )
    parser.add_argument("--model-path", default=None, help="SFT actor checkpoint path.")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Local LeRobot dataset path. Used as OpenPI repo_id unless --repo-id is set.",
    )
    parser.add_argument("--repo-id", default=None, help="Explicit OpenPI repo_id/path.")
    parser.add_argument("--norm-stats-path", default=None, help="Explicit norm stats path.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--env-split", choices=("train", "eval"), default="eval")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="Optional ManiSkill reset episode_id. If omitted, uses the env config reset behavior.",
    )
    parser.add_argument(
        "--save-images",
        default=None,
        help="Optional directory for raw/transformed train/env image dumps.",
    )
    parser.add_argument(
        "--keep-compile",
        action="store_true",
        help="Keep model torch.compile settings. Disabled by default for faster debug startup.",
    )
    parser.add_argument(
        "--skip-gt-action-replay",
        action="store_true",
        help="Skip replaying the dataset GT action chunk in the env.",
    )
    parser.add_argument(
        "--scan-action-samples",
        type=int,
        default=0,
        help="Optionally scan this many source samples for action/gripper ranges.",
    )
    parser.add_argument(
        "--full-gt-replay",
        action="store_true",
        help="Replay one GT action per dataset frame for a full demo rollout.",
    )
    parser.add_argument(
        "--full-gt-replay-max-steps",
        type=int,
        default=500,
        help="Maximum env steps for --full-gt-replay.",
    )
    parser.add_argument(
        "--full-gt-replay-log-every",
        type=int,
        default=25,
        help="Print a full GT replay progress row every N env steps.",
    )
    parser.add_argument(
        "--full-gt-replay-action-index",
        type=int,
        default=0,
        help="Which action inside each dataset action chunk to execute during full GT replay.",
    )
    parser.add_argument(
        "--full-gt-replay-action-candidate",
        default="dataset_xyz_euler_to_rotvec_gripper_raw",
        help="Action conversion candidate to execute during full GT replay.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides. Direct path args above are safer for absolute paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.env_index < 0 or args.env_index >= args.num_envs:
        raise ValueError("--env-index must be in [0, --num-envs).")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = _compose_cfg(args)
    _maybe_set_lerobot_home(cfg)

    _print_section("Resolved Debug Config")
    print(f"config_name={args.config_name}")
    print(f"model_path={cfg.actor.model.model_path}")
    data_kwargs = _openpi_data_kwargs(cfg)
    print(f"openpi_data={OmegaConf.to_container(data_kwargs, resolve=True) if data_kwargs is not None else None}")
    print(f"env_split={args.env_split}")
    print(
        "env_task="
        f"{cfg.env[args.env_split].init_params.id} "
        "control_mode="
        f"{cfg.env[args.env_split].init_params.control_mode}"
    )

    source_sample, train_sample, _, source_dataset = _load_train_samples(
        cfg,
        args.sample_index,
    )

    _print_section("Source Training Sample")
    _summarize_tree(source_sample, "source")

    _print_section("Transformed Training Sample")
    _summarize_tree(train_sample, "train_sample")
    _debug_dataset_action_scan(source_dataset, args.scan_action_samples)

    model = get_model(cfg.actor.model, torch_dtype=None)
    model.eval()

    _print_section("Model Transform Config")
    print(f"model.config.config_name={getattr(model.config, 'config_name', None)}")
    print(f"model.config.action_chunk={getattr(model.config, 'action_chunk', None)}")
    print(f"model.config.action_horizon={getattr(model.config, 'action_horizon', None)}")
    print(f"model.config.action_dim={getattr(model.config, 'action_dim', None)}")
    print(f"model.config.action_env_dim={getattr(model.config, 'action_env_dim', None)}")

    env = _build_env(cfg, args)
    try:
        env_obs, env_infos, raw_env_obs, raw_debug = _reset_env_with_raw(env, args)
        live_debug = _extract_live_env_debug_fields(env)

        _print_section("Raw Env Observation")
        _summarize_tree(env_obs, "env_obs")
        if hasattr(env, "reset_state_ids"):
            _stats("env.reset_state_ids", env.reset_state_ids)

        _print_raw_env_pose_debug(
            source_sample=source_sample,
            raw_debug=raw_debug,
            live_debug=live_debug,
            env_index=args.env_index,
        )

        _print_section("Env Reset Infos")
        _summarize_tree(env_infos, "env_infos")

        to_process_obs, env_transformed = _transform_env_obs(model, env_obs)
        env_transformed_sample = _select_batch(env_transformed, args.env_index)

        _print_section("Env Policy Input Before OpenPI Transform")
        _summarize_tree(to_process_obs, "policy_input")

        _print_section("Env Observation After OpenPI Transform")
        _summarize_tree(env_transformed_sample, "env_transformed_sample")

        _compare_core_fields(
            source_sample=source_sample,
            train_sample=train_sample,
            env_obs=env_obs,
            env_transformed_sample=env_transformed_sample,
            env_index=args.env_index,
        )

        _maybe_save_images(
            args.save_images,
            source_sample,
            train_sample,
            env_obs,
            env_transformed_sample,
            args.env_index,
        )

        _debug_policy_action_on_env_obs(
            model=model,
            env_obs=env_obs,
            source_sample=source_sample,
            cfg=cfg,
            args=args,
        )

        if not args.skip_gt_action_replay:
            _debug_gt_action_replay(
                env=env,
                args=args,
                source_dataset=source_dataset,
                sample_index=args.sample_index,
                source_sample=source_sample,
            )
        if args.full_gt_replay:
            _debug_full_gt_action_replay(
                env=env,
                args=args,
                source_dataset=source_dataset,
                sample_index=args.sample_index,
                source_sample=source_sample,
            )
    finally:
        raw_env = getattr(env, "env", None)
        if raw_env is not None and hasattr(raw_env, "close"):
            raw_env.close()


if __name__ == "__main__":
    main()
