#!/usr/bin/env python3
"""Collect PegInsertionSide-v1 demos for ``pi05_rlt_maniskill``.

The collector uses ManiSkill's official Panda motion-planning solution as a
reference trajectory generator, then tracks that reference in a fresh
``pd_ee_delta_pose`` environment. Only the actions actually sent to the
``pd_ee_delta_pose`` controller are saved.

    image, wrist_image, state, actions, task

State/action semantics:
    state[0:3]   = TCP position in robot root frame, meters
    state[3:6]   = TCP orientation as rotation vector, robot root frame
    state[6:8]   = Panda finger qpos values
    actions[0:3] = normalized pd_ee_delta_pose position command
    actions[3:6] = normalized pd_ee_delta_pose rotation-vector command
    actions[6]   = normalized gripper command, +1=open and -1=close

The action is env-native for ManiSkill's Panda ``pd_ee_delta_pose`` controller,
so RLinf's ``rlt_maniskill`` action wrapper only clips it to [-1, 1].
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used in minimal local shells.
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total: int, desc: str = ""):
            self.total = total
            self.desc = desc
            self.count = 0

        def update(self, n: int = 1) -> None:
            self.count += n
            print(f"{self.desc}: {self.count}/{self.total}", flush=True)

        def close(self) -> None:
            pass


def _bootstrap_repo_paths() -> Path:
    script_path = Path(__file__).resolve()
    rlinf_root = script_path.parents[2]
    repo_root = rlinf_root.parent

    for candidate in (rlinf_root, repo_root / "openpi-RLT" / "src"):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    os.environ.setdefault("EMBODIED_PATH", str(script_path.parent))
    return rlinf_root


RLINF_ROOT = _bootstrap_repo_paths()


LOG = logging.getLogger("collect_maniskill_peg_lerobot")

ENV_ID = "PegInsertionSide-v1"
STATE_DIM = 8
ACTION_DIM = 7
DEFAULT_TASK = "insert the peg in the hole"

MAIN_CAMERA_CANDIDATES = ("base_camera", "3rd_view_camera")
WRIST_CAMERA_CANDIDATES = ("hand_camera",)

SOLVER_MODULE_CANDIDATES = (
    "mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side",
    "mani_skill.examples.motionplanning.panda.peg_insertion_side",
)


@dataclasses.dataclass(frozen=True)
class FrameRecord:
    obs: dict[str, Any]
    state: np.ndarray
    tcp_pos: np.ndarray
    tcp_quat_wxyz: np.ndarray


@dataclasses.dataclass(frozen=True)
class CandidateEpisode:
    frames: list[dict[str, Any]]
    records: list[FrameRecord]
    actions: list[np.ndarray]
    final_success: bool


def _to_numpy(value: Any, *, squeeze_env_dim: bool = True) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if squeeze_env_dim and arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _bool_scalar(value: Any) -> bool:
    arr = _to_numpy(value)
    if arr.size == 0:
        return False
    return bool(np.asarray(arr).reshape(-1)[0])


def _solver_success(result: Any) -> bool:
    if isinstance(result, dict):
        if "success" in result:
            return _bool_scalar(result["success"])
        return False
    if isinstance(result, tuple):
        for item in reversed(result):
            if isinstance(item, dict) and "success" in item:
                return _bool_scalar(item["success"])
            if isinstance(item, (bool, np.bool_)) or hasattr(item, "detach"):
                return _bool_scalar(item)
    return _bool_scalar(result)


def _import_solver():
    errors: list[str] = []
    for module_name in SOLVER_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        solve = getattr(module, "solve", None)
        if solve is not None:
            return solve, module_name
        errors.append(f"{module_name}: missing solve()")

    raise ImportError(
        "Could not import ManiSkill PegInsertionSide motion-planning solver. "
        "Tried:\n  " + "\n  ".join(errors)
    )


def _missing_dep_error(package: str, install_hint: str) -> RuntimeError:
    return RuntimeError(
        f"Missing runtime dependency '{package}'. Install it in the RLinf/ManiSkill "
        f"environment before collecting data. Suggested package: {install_hint}"
    )


def _quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm
    if quat[0] < 0:
        quat = -quat

    w = np.clip(quat[0], -1.0, 1.0)
    xyz = quat[1:]
    sin_half_angle = np.linalg.norm(xyz)
    if sin_half_angle < 1e-8:
        return (2.0 * xyz).astype(np.float32)

    angle = 2.0 * np.arctan2(sin_half_angle, w)
    return (xyz / sin_half_angle * angle).astype(np.float32)


def _normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _normalize_quat_wxyz(lhs)
    w2, x2, y2, z2 = _normalize_quat_wxyz(rhs)
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_inverse_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(quat_wxyz)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def _quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Invert RLinf's XYZ-Euler-to-quaternion convention in action_utils.py."""
    w, x, y, z = _normalize_quat_wxyz(quat_wxyz)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _rotation_delta_euler_xyz(
    curr_quat_wxyz: np.ndarray, next_quat_wxyz: np.ndarray
) -> np.ndarray:
    delta = _quat_multiply_wxyz(next_quat_wxyz, _quat_inverse_wxyz(curr_quat_wxyz))
    return _quat_wxyz_to_euler_xyz(delta)


def _quat_wxyz_to_rotvec_no_canonical(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat_wxyz)
    w = np.clip(quat[0], -1.0, 1.0)
    xyz = quat[1:]
    sin_half_angle = np.linalg.norm(xyz)
    if sin_half_angle < 1e-8:
        return (2.0 * xyz).astype(np.float32)

    angle = 2.0 * np.arctan2(sin_half_angle, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return (xyz / sin_half_angle * angle).astype(np.float32)


def _rotation_delta_rotvec(
    curr_quat_wxyz: np.ndarray,
    target_quat_wxyz: np.ndarray,
) -> np.ndarray:
    delta = _quat_multiply_wxyz(target_quat_wxyz, _quat_inverse_wxyz(curr_quat_wxyz))
    return _quat_wxyz_to_rotvec_no_canonical(delta)


def _pose_from_env(env: Any) -> tuple[np.ndarray, np.ndarray] | None:
    base_env = getattr(env, "unwrapped", env)
    agent = getattr(base_env, "agent", None)
    if agent is None:
        return None

    pose = getattr(agent, "ee_pose_at_robot_base", None)
    if pose is None:
        robot = getattr(agent, "robot", None)
        tcp = getattr(agent, "tcp", None)
        robot_pose = getattr(robot, "pose", None)
        tcp_pose = getattr(tcp, "pose", None)
        if robot_pose is None or tcp_pose is None:
            return None
        try:
            pose = robot_pose.inv() * tcp_pose
        except Exception:
            return None

    raw_pose = getattr(pose, "raw_pose", pose)
    pose_np = _to_numpy(raw_pose).astype(np.float32)
    if pose_np.shape[-1] != 7:
        return None
    return pose_np[:3], pose_np[3:]


def _pose_from_obs(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    for container_key in ("extra", "agent"):
        container = obs.get(container_key, {})
        if not isinstance(container, dict) or "tcp_pose" not in container:
            continue
        pose = _to_numpy(container["tcp_pose"]).astype(np.float32)
        if pose.shape[-1] == 7:
            return pose[:3], pose[3:]

    raise KeyError(
        "Could not find TCP pose. Expected env.agent.ee_pose_at_robot_base, "
        "obs['extra']['tcp_pose'], or obs['agent']['tcp_pose']."
    )


def _extract_record(env: Any, obs: dict[str, Any]) -> FrameRecord:
    pose = _pose_from_env(env)
    if pose is None:
        pose = _pose_from_obs(obs)
    tcp_pos, tcp_quat_wxyz = pose

    qpos = _to_numpy(obs["agent"]["qpos"]).astype(np.float32)
    if qpos.shape[0] < 9:
        raise ValueError(f"Expected Panda qpos with at least 9 values, got {qpos.shape}")

    state = np.concatenate(
        [
            tcp_pos.astype(np.float32),
            _quat_wxyz_to_rotvec(tcp_quat_wxyz),
            qpos[7:9].astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"Expected {STATE_DIM}D state, got {state.shape}")

    return FrameRecord(
        obs=obs,
        state=state,
        tcp_pos=tcp_pos.astype(np.float32),
        tcp_quat_wxyz=tcp_quat_wxyz.astype(np.float32),
    )


def _camera_image(obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    sensors = obs.get("sensor_data", {})
    sensor = sensors.get(camera_name)
    if not isinstance(sensor, dict) or "rgb" not in sensor:
        return None
    image = _to_numpy(sensor["rgb"]).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Camera {camera_name} produced invalid RGB shape {image.shape}")
    return image


def _available_rgb_cameras(obs: dict[str, Any]) -> list[str]:
    sensors = obs.get("sensor_data", {})
    names: list[str] = []
    for name, sensor in sensors.items():
        if isinstance(sensor, dict) and sensor.get("rgb") is not None:
            names.append(name)
    return names


def _select_camera(
    obs: dict[str, Any],
    requested: str,
    candidates: tuple[str, ...],
    role: str,
) -> str:
    if requested:
        if _camera_image(obs, requested) is None:
            raise ValueError(
                f"Requested {role} camera '{requested}' is unavailable. "
                f"Available RGB cameras: {_available_rgb_cameras(obs)}"
            )
        return requested

    for camera_name in candidates:
        if _camera_image(obs, camera_name) is not None:
            return camera_name

    raise ValueError(
        f"No {role} camera found. Tried {candidates}; "
        f"available RGB cameras: {_available_rgb_cameras(obs)}"
    )


def _make_action(
    curr: FrameRecord,
    nxt: FrameRecord,
    *,
    gripper_open_threshold: float,
) -> np.ndarray:
    delta_pos = (nxt.tcp_pos - curr.tcp_pos).astype(np.float32)
    delta_euler_xyz = _rotation_delta_euler_xyz(curr.tcp_quat_wxyz, nxt.tcp_quat_wxyz)
    finger_mean = float(np.mean(nxt.state[6:8]))
    gripper = np.asarray(
        [1.0 if finger_mean > gripper_open_threshold else 0.0], dtype=np.float32
    )

    action = np.concatenate([delta_pos, delta_euler_xyz, gripper], axis=0).astype(
        np.float32
    )
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"Expected {ACTION_DIM}D action, got {action.shape}")
    return action


def _make_tracking_action(
    curr: FrameRecord,
    target: FrameRecord,
    *,
    pos_scale: float,
    rot_scale: float,
    max_action: float,
    gripper_open_threshold: float,
) -> np.ndarray:
    delta_pos = (target.tcp_pos - curr.tcp_pos).astype(np.float32) * float(pos_scale)
    delta_rot = _rotation_delta_rotvec(
        curr.tcp_quat_wxyz,
        target.tcp_quat_wxyz,
    ) * float(rot_scale)
    finger_mean = float(np.mean(target.state[6:8]))
    gripper = 1.0 if finger_mean > gripper_open_threshold else -1.0
    action = np.concatenate(
        [
            delta_pos,
            delta_rot,
            np.asarray([gripper], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return np.clip(action, -float(max_action), float(max_action)).astype(np.float32)


def _episode_to_frames(
    records: list[FrameRecord],
    *,
    task: str,
    main_camera: str,
    wrist_camera: str,
    gripper_open_threshold: float,
    actions: list[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    if len(records) < 2:
        raise ValueError("Need at least two observations to build one action")
    if actions is not None and len(actions) != len(records) - 1:
        raise ValueError(
            f"Expected {len(records) - 1} actions for {len(records)} records, got {len(actions)}"
        )

    frames: list[dict[str, Any]] = []
    for idx, (curr, nxt) in enumerate(zip(records[:-1], records[1:])):
        image = _camera_image(curr.obs, main_camera)
        if image is None:
            raise ValueError(f"Main camera '{main_camera}' missing from observation")

        wrist_image = _camera_image(curr.obs, wrist_camera)
        if wrist_image is None:
            raise ValueError(f"Wrist camera '{wrist_camera}' missing from observation")

        frames.append(
            {
                "image": image,
                "wrist_image": wrist_image,
                "state": curr.state,
                "actions": (
                    np.asarray(actions[idx], dtype=np.float32)
                    if actions is not None
                    else _make_action(
                        curr,
                        nxt,
                        gripper_open_threshold=gripper_open_threshold,
                    )
                ),
                "task": task,
            }
        )

    return frames


def _video_output_dir(repo_id: str, requested_video_dir: str) -> Path:
    if requested_video_dir:
        return Path(requested_video_dir).expanduser()

    dataset_path = _resolve_output_path(repo_id)
    return dataset_path.with_name(f"{dataset_path.name}_videos")


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    pad = np.zeros((height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
    return np.concatenate([image, pad], axis=0)


def _video_view(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Expected video image rank 3, got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected video image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _make_video_frame(frame: dict[str, Any]) -> np.ndarray:
    main = _video_view(frame["image"])
    wrist = _video_view(frame["wrist_image"])
    height = max(main.shape[0], wrist.shape[0])
    main = _pad_to_height(main, height)
    wrist = _pad_to_height(wrist, height)
    gap = np.full((height, 4, 3), 32, dtype=np.uint8)
    return np.concatenate([main, gap, wrist], axis=1)


def _write_episode_video(
    frames: list[dict[str, Any]],
    *,
    video_dir: Path,
    episode_index: int,
    seed: int,
    fps: int,
) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"episode_{episode_index:06d}_seed_{seed:06d}.mp4"
    video_frames = np.stack([_make_video_frame(frame) for frame in frames], axis=0)

    try:
        import imageio.v3 as iio

        iio.imwrite(video_path, video_frames, fps=fps)
        return
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        LOG.warning("imageio.v3 failed to write %s: %s", video_path, exc)

    try:
        import imageio

        imageio.mimsave(video_path, list(video_frames), fps=fps)
    except ImportError as exc:
        raise _missing_dep_error("imageio", "imageio imageio-ffmpeg") from exc


def _run_solver_reference(
    *,
    env: Any,
    solve: Any,
    seed: int,
) -> list[FrameRecord] | None:
    records: list[FrameRecord] = []

    orig_reset = env.reset
    orig_step = env.step

    def reset_hook(*hook_args, **hook_kwargs):
        out = orig_reset(*hook_args, **hook_kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        records.clear()
        records.append(_extract_record(env, obs))
        return out

    def step_hook(action, *hook_args, **hook_kwargs):
        out = orig_step(action, *hook_args, **hook_kwargs)
        obs = out[0]
        records.append(_extract_record(env, obs))
        return out

    env.reset = reset_hook  # type: ignore[method-assign]
    env.step = step_hook  # type: ignore[method-assign]
    try:
        result = solve(env, seed=seed, debug=False, vis=False)
        if not _solver_success(result) or len(records) < 2:
            return None
        return list(records)
    finally:
        env.reset = orig_reset  # type: ignore[method-assign]
        env.step = orig_step  # type: ignore[method-assign]


def _track_reference_with_ee_controller(
    *,
    env: Any,
    seed: int,
    reference_records: list[FrameRecord],
    args: argparse.Namespace,
    task: str,
    main_camera: str,
    wrist_camera: str,
) -> CandidateEpisode | None:
    obs, _ = env.reset(seed=seed)
    records = [_extract_record(env, obs)]
    actions: list[np.ndarray] = []
    max_state_error = 0.0
    last_info: dict[str, Any] = {}
    last_terminated = False
    last_truncated = False
    max_steps = args.track_max_steps or args.max_episode_steps
    track_steps = min(len(reference_records) - 1, max_steps)

    for step_idx in range(track_steps):
        curr = records[-1]
        target = reference_records[min(step_idx + 1, len(reference_records) - 1)]
        action = _make_tracking_action(
            curr,
            target,
            pos_scale=args.pos_action_scale,
            rot_scale=args.rot_action_scale,
            max_action=args.max_action,
            gripper_open_threshold=args.gripper_open_threshold,
        )
        obs, _reward, terminated, truncated, info = env.step(action)
        last_info = info
        last_terminated = _bool_scalar(terminated)
        last_truncated = _bool_scalar(truncated)
        actions.append(action)
        records.append(_extract_record(env, obs))

        ref_state = reference_records[min(step_idx + 1, len(reference_records) - 1)].state
        state_error = np.max(np.abs(records[-1].state - ref_state))
        max_state_error = max(max_state_error, float(state_error))

        if _bool_scalar(info.get("success", False)):
            break
        if last_terminated or last_truncated:
            break

    if len(records) < 2 or len(actions) != len(records) - 1:
        LOG.info("Rejecting seed %d: tracking produced no valid episode", seed)
        return None
    if max_state_error > float(args.track_state_error_threshold):
        LOG.info(
            "Rejecting seed %d: max tracked state error %.6g > %.6g",
            seed,
            max_state_error,
            args.track_state_error_threshold,
        )
        return None
    if not _bool_scalar(last_info.get("success", False)):
        LOG.info(
            "Rejecting seed %d: EE tracking did not succeed "
            "(terminated=%s truncated=%s max_state_error=%.6g last_success=%s)",
            seed,
            last_terminated,
            last_truncated,
            max_state_error,
            _bool_scalar(last_info.get("success", False)),
        )
        return None

    frames = _episode_to_frames(
        records,
        task=task,
        main_camera=main_camera,
        wrist_camera=wrist_camera,
        gripper_open_threshold=args.gripper_open_threshold,
        actions=actions,
    )
    return CandidateEpisode(
        frames=frames,
        records=records,
        actions=actions,
        final_success=True,
    )


def _resolve_output_path(repo_id: str) -> Path:
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    except ImportError as exc:
        raise _missing_dep_error("lerobot", "lerobot") from exc

    repo_path = Path(repo_id).expanduser()
    if repo_path.is_absolute():
        return repo_path
    return HF_LEROBOT_HOME / repo_id


def _create_dataset(
    *,
    repo_id: str,
    image_shape: tuple[int, int, int],
    wrist_image_shape: tuple[int, int, int],
    fps: int,
    image_writer_threads: int,
    image_writer_processes: int,
):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise _missing_dep_error("lerobot", "lerobot") from exc

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": wrist_image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def _build_env(args: argparse.Namespace, *, control_mode: str):
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise _missing_dep_error("gymnasium", "gymnasium") from exc
    try:
        import mani_skill.envs  # noqa: F401
    except ImportError as exc:
        raise _missing_dep_error("mani_skill", "mani_skill") from exc

    sim_freq = max(args.sim_freq, args.control_freq * 8)
    sim_freq -= sim_freq % args.control_freq

    env_kwargs: dict[str, Any] = {
        "id": ENV_ID,
        "obs_mode": "rgb",
        "control_mode": control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "rgb_array",
        "sim_backend": args.sim_backend,
        "sim_config": {"sim_freq": sim_freq, "control_freq": args.control_freq},
        "sensor_configs": {
            "shader_pack": args.shader_pack,
            "width": args.image_width,
            "height": args.image_height,
        },
        "max_episode_steps": args.max_episode_steps,
    }
    if args.robot_uids:
        env_kwargs["robot_uids"] = args.robot_uids

    return gym.make(**env_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect successful ManiSkill PegInsertionSide-v1 motion-planning "
            "episodes as a LeRobot dataset compatible with pi05_rlt_maniskill."
        )
    )
    parser.add_argument("--repo-id", default="local/maniskill_peginsertionside_rlt_200")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--main-camera", default="")
    parser.add_argument("--wrist-camera", default="")
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-freq", type=int, default=100)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--shader-pack", default="default")
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--robot-uids", default="")
    parser.add_argument(
        "--solver-control-mode",
        default="pd_joint_pos",
        help="Keep this as pd_joint_pos for ManiSkill's Panda motion-planning solver.",
    )
    parser.add_argument(
        "--target-control-mode",
        default="pd_ee_delta_pose",
        help="Controller used for the saved dataset actions.",
    )
    parser.add_argument(
        "--pos-action-scale",
        type=float,
        default=10.0,
        help="Tracking gain from TCP position error in meters to normalized EE action.",
    )
    parser.add_argument(
        "--rot-action-scale",
        type=float,
        default=10.0,
        help="Tracking gain from TCP rotation-vector error in radians to normalized EE action.",
    )
    parser.add_argument(
        "--max-action",
        type=float,
        default=1.0,
        help="Absolute clip for saved normalized EE actions.",
    )
    parser.add_argument(
        "--track-max-steps",
        type=int,
        default=0,
        help="Max EE tracking steps per candidate. Defaults to --max-episode-steps.",
    )
    parser.add_argument(
        "--track-state-error-threshold",
        type=float,
        default=10.0,
        help=(
            "Reject tracked episodes whose max state error versus the reference "
            "exceeds this value. Large default only rejects explosions."
        ),
    )
    parser.add_argument("--gripper-open-threshold", type=float, default=0.02)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=4)
    parser.add_argument(
        "--save-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one side-by-side mp4 per successful episode for visual inspection.",
    )
    parser.add_argument(
        "--video-dir",
        default="",
        help="Video output directory. Defaults to <dataset_path>_videos.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    output_path = _resolve_output_path(args.repo_id)
    video_dir = _video_output_dir(args.repo_id, args.video_dir)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset already exists at {output_path}. Pass --overwrite to replace it."
            )
        LOG.info("Removing existing dataset at %s", output_path)
        shutil.rmtree(output_path)
    if args.save_videos and video_dir.exists() and args.overwrite:
        LOG.info("Removing existing video directory at %s", video_dir)
        shutil.rmtree(video_dir)

    solve, solver_module = _import_solver()
    LOG.info("Using ManiSkill solver: %s", solver_module)

    solver_env = _build_env(args, control_mode=args.solver_control_mode)
    target_env = _build_env(args, control_mode=args.target_control_mode)

    dataset = None
    main_camera = ""
    wrist_camera = ""
    saved = 0
    attempts = 0
    solver_failures = 0
    tracking_failures = 0
    conversion_failures = 0
    pbar = tqdm(total=args.num_episodes, desc="Successful episodes")

    try:
        while saved < args.num_episodes and attempts < args.max_attempts:
            episode_seed = args.seed + attempts
            attempts += 1

            try:
                reference_records = _run_solver_reference(
                    env=solver_env,
                    solve=solve,
                    seed=episode_seed,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Solver failed on seed %d: %s", episode_seed, exc)
                solver_failures += 1
                continue

            if reference_records is None:
                solver_failures += 1
                continue

            try:
                if not main_camera:
                    main_camera = _select_camera(
                        reference_records[0].obs,
                        args.main_camera,
                        MAIN_CAMERA_CANDIDATES,
                        "main",
                    )
                    wrist_camera = _select_camera(
                        reference_records[0].obs,
                        args.wrist_camera,
                        WRIST_CAMERA_CANDIDATES,
                        "wrist",
                    )
                    LOG.info(
                        "Selected cameras: image=%s, wrist_image=%s",
                        main_camera,
                        wrist_camera,
                    )

                candidate = _track_reference_with_ee_controller(
                    env=target_env,
                    seed=episode_seed,
                    reference_records=reference_records,
                    args=args,
                    task=args.task,
                    main_camera=main_camera,
                    wrist_camera=wrist_camera,
                )
                if candidate is None:
                    tracking_failures += 1
                    continue
                frames = candidate.frames
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Conversion failed on seed %d: %s", episode_seed, exc)
                conversion_failures += 1
                continue

            if dataset is None:
                dataset = _create_dataset(
                    repo_id=args.repo_id,
                    image_shape=tuple(frames[0]["image"].shape),
                    wrist_image_shape=tuple(frames[0]["wrist_image"].shape),
                    fps=args.fps,
                    image_writer_threads=args.image_writer_threads,
                    image_writer_processes=args.image_writer_processes,
                )

            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            if args.save_videos:
                _write_episode_video(
                    frames,
                    video_dir=video_dir,
                    episode_index=saved,
                    seed=episode_seed,
                    fps=args.fps,
                )

            saved += 1
            pbar.update(1)

    finally:
        pbar.close()
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        solver_env.close()
        target_env.close()

    if saved < args.num_episodes:
        raise RuntimeError(
            f"Only saved {saved}/{args.num_episodes} successful episodes after "
            f"{attempts} attempts. "
            f"solver_failures={solver_failures}, "
            f"tracking_failures={tracking_failures}, "
            f"conversion_failures={conversion_failures}. "
            "Increase --max-attempts or inspect the logged rejection reasons."
        )

    LOG.info("Saved %d successful episodes after %d attempts to %s", saved, attempts, output_path)


if __name__ == "__main__":
    main()
