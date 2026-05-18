# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, OrderedDict, Union

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import put_info_on_image, tile_images
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

__all__ = ["ManiskillEnv"]


def extract_termination_from_info(info, num_envs, device):
    if "success" in info:
        if "fail" in info:
            terminated = torch.logical_or(info["success"], info["fail"])
        else:
            terminated = info["success"].clone()
    else:
        if "fail" in info:
            terminated = info["fail"].clone()
        else:
            terminated = torch.zeros(num_envs, dtype=bool, device=device)
    return terminated


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _quat_xyzw_to_rpy(quat_xyzw):
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


def _extract_tcp_pose(obs):
    extra = obs.get("extra", {})
    tcp_pose = extra.get("tcp_pose")
    if tcp_pose is not None:
        pose = _to_numpy(tcp_pose).astype(np.float32)
        return pose[:3], pose[3:]

    agent = obs.get("agent", {})
    if "tcp_pose" in agent:
        pose = _to_numpy(agent["tcp_pose"]).astype(np.float32)
        return pose[:3], pose[3:]

    raise KeyError(
        "Could not find TCP pose in ManiSkill observation. "
        "Expected obs['extra']['tcp_pose'] or obs['agent']['tcp_pose']."
    )


def _extract_rlt_state(raw_obs, device):
    pos, quat_xyzw = _extract_tcp_pose(raw_obs)
    rpy = _quat_xyzw_to_rpy(quat_xyzw)
    qpos = _to_numpy(raw_obs["agent"]["qpos"]).astype(np.float32)
    fingers = qpos[7:9]
    state = np.concatenate([pos.astype(np.float32), rpy, fingers], axis=0)
    return torch.as_tensor(state, device=device, dtype=torch.float32)


class ManiskillEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.use_full_state = bool(getattr(cfg, "use_full_state", False))
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg
        self.task_id = getattr(cfg.init_params, "id", None)
        self._is_peg_insertion_side = self.task_id == "PegInsertionSide-v1"

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        self.env: BaseEnv = gym.make(**env_args)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )  # [B, ]
        self.record_metrics = record_metrics
        self._is_start = True
        self._init_reset_state_ids()
        if self._is_peg_insertion_side:
            self.info_logging_keys = [
                "is_grasped_current",
                "consecutive_grasp_once",
                "prealign_once",
                "partial_insert_once",
                "success",
                "peg_head_hole_x",
                "peg_head_goal_yz_dist",
                "peg_body_goal_yz_dist",
            ]
            self._init_peg_insertion_event_state()
        else:
            self.info_logging_keys = [
                "is_src_obj_grasped",
                "consecutive_grasp",
                "success",
            ]
        self._show_goal_site_visual()
        if self.record_metrics:
            self._init_metrics()

    @property
    def total_num_group_envs(self):
        if hasattr(self.env.unwrapped, "total_num_trials"):
            return self.env.unwrapped.total_num_trials
        if hasattr(self.env, "xyz_configs") and hasattr(self.env, "quat_configs"):
            return len(self.env.xyz_configs) * len(self.env.quat_configs)
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def num_envs(self):
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def elapsed_steps(self):
        return self.env.unwrapped.elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def instruction(self):
        env = self.env.unwrapped
        if hasattr(env, "get_language_instruction"):
            instruction = env.get_language_instruction()
            if instruction is not None:
                return instruction

        for attr in (
            "task_descriptions",
            "task_description",
            "task_prompt",
            "instruction",
        ):
            if not hasattr(env, attr):
                continue
            instruction = getattr(env, attr)
            if instruction is None:
                continue
            if isinstance(instruction, str):
                return [instruction for _ in range(self.num_envs)]
            return instruction

        if self._is_peg_insertion_side:
            return ["insert the peg into the hole" for _ in range(self.num_envs)]
        return ["" for _ in range(self.num_envs)]

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _show_goal_site_visual(self):
        """Keep ManiSkill goal-site visualization visible for reward-model RGB input."""
        if not hasattr(self.env.unwrapped, "goal_site"):
            return

        goal_site = self.env.unwrapped.goal_site
        if hasattr(self.env.unwrapped, "_hidden_objects"):
            while goal_site in self.env.unwrapped._hidden_objects:
                self.env.unwrapped._hidden_objects.remove(goal_site)
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

    def _wrap_obs(self, raw_obs, infos=None):
        wrap_obs_mode = getattr(self.cfg, "wrap_obs_mode", "default")
        if wrap_obs_mode == "raw":
            assert infos is not None
            return infos["extracted_obs"]

        if wrap_obs_mode == "simple":
            if self.env.unwrapped.obs_mode == "state":
                return {"states": raw_obs}
            elif self.env.unwrapped.obs_mode == "rgb":
                sensor_data = raw_obs.pop("sensor_data")
                raw_obs.pop("sensor_param")
                if self.use_full_state:
                    state = self._get_full_state_obs()
                else:
                    state = common.flatten_state_dict(
                        raw_obs, use_torch=True, device=self.device
                    )

                main_images = sensor_data["base_camera"]["rgb"]
                sorted_images = OrderedDict(sorted(sensor_data.items()))
                sorted_images.pop("base_camera")
                extra_view_images = (
                    torch.stack([v["rgb"] for v in sorted_images.values()], dim=1)
                    if sorted_images
                    else None
                )
                return {
                    "main_images": main_images,
                    "extra_view_images": extra_view_images,
                    "states": state,
                }

        if wrap_obs_mode == "rlt_openpi":
            if self.env.unwrapped.obs_mode != "rgb":
                raise ValueError(
                    "wrap_obs_mode='rlt_openpi' requires ManiSkill obs_mode='rgb'."
                )
            sensor_data = raw_obs.pop("sensor_data")
            raw_obs.pop("sensor_param")

            main_images = sensor_data["base_camera"]["rgb"]
            wrist_images = (
                sensor_data["hand_camera"]["rgb"]
                if "hand_camera" in sensor_data
                else None
            )
            extra_view_images = None
            if "right_camera" in sensor_data:
                extra_view_images = sensor_data["right_camera"]["rgb"].unsqueeze(1)

            if infos is not None and "prompt" in infos:
                task_descriptions = infos["prompt"]
            else:
                task_descriptions = self.instruction

            return {
                "main_images": main_images,
                "wrist_images": wrist_images,
                "extra_view_images": extra_view_images,
                "states": torch.stack(
                    [
                        _extract_rlt_state(
                            {
                                "agent": {
                                    "qpos": raw_obs["agent"]["qpos"][i],
                                    **(
                                        {"tcp_pose": raw_obs["agent"]["tcp_pose"][i]}
                                        if isinstance(raw_obs["agent"], dict)
                                        and "tcp_pose" in raw_obs["agent"]
                                        else {}
                                    ),
                                },
                                "extra": (
                                    {"tcp_pose": raw_obs["extra"]["tcp_pose"][i]}
                                    if isinstance(raw_obs.get("extra"), dict)
                                    and "tcp_pose" in raw_obs["extra"]
                                    else {}
                                ),
                            },
                            self.device,
                        )
                        for i in range(main_images.shape[0])
                    ],
                    dim=0,
                ),
                "task_descriptions": task_descriptions,
            }

        # Default
        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(
            torch.uint8
        )  # [B, H, W, C]
        proprioception: torch.Tensor = self.env.unwrapped.agent.robot.get_qpos().to(
            obs_image.device, dtype=torch.float32
        )
        return {
            "main_images": obs_image,
            "states": proprioception,
            "task_descriptions": self.instruction,
        }

    def _get_full_state_obs(self):
        base_env = self.env.unwrapped
        mode_attr = "_obs_mode" if hasattr(base_env, "_obs_mode") else "obs_mode"
        original_mode = getattr(base_env, mode_attr)
        setattr(base_env, mode_attr, "state")
        try:
            state_obs = base_env.get_obs()
        finally:
            setattr(base_env, mode_attr, original_mode)

        if isinstance(state_obs, dict):
            return common.flatten_state_dict(
                state_obs, use_torch=True, device=self.device
            )
        return state_obs

    def _init_peg_insertion_event_state(self):
        self.peg_grasp_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        self.peg_consecutive_grasp_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.peg_prealign_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.peg_partial_insert_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.peg_success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _reset_peg_insertion_event_state(self, env_idx=None):
        if not self._is_peg_insertion_side:
            return

        if env_idx is None:
            self.peg_grasp_count.zero_()
            self.peg_consecutive_grasp_once.zero_()
            self.peg_prealign_once.zero_()
            self.peg_partial_insert_once.zero_()
            self.peg_success_once.zero_()
            return

        self.peg_grasp_count[env_idx] = 0
        self.peg_consecutive_grasp_once[env_idx] = False
        self.peg_prealign_once[env_idx] = False
        self.peg_partial_insert_once[env_idx] = False
        self.peg_success_once[env_idx] = False

    def _augment_peg_insertion_info(self, infos):
        if not self._is_peg_insertion_side:
            return infos

        env = self.env.unwrapped
        peg_head_pos_at_hole = infos.get("peg_head_pos_at_hole")
        if peg_head_pos_at_hole is None:
            peg_head_pos_at_hole = (env.box_hole_pose.inv() * env.peg_head_pose).p
        peg_head_pos_at_hole = peg_head_pos_at_hole.to(self.device, dtype=torch.float32)

        peg_head_wrt_goal = env.goal_pose.inv() * env.peg_head_pose
        peg_wrt_goal = env.goal_pose.inv() * env.peg.pose
        peg_head_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], dim=1
        ).to(torch.float32)
        peg_body_goal_yz_dist = torch.linalg.norm(
            peg_wrt_goal.p[:, 1:], dim=1
        ).to(torch.float32)

        is_grasped_current = env.agent.is_grasping(env.peg, max_angle=20)
        self.peg_grasp_count = torch.where(
            is_grasped_current,
            self.peg_grasp_count + 1,
            torch.zeros_like(self.peg_grasp_count),
        )
        consecutive_grasp_current = self.peg_grasp_count >= 5

        prealigned_current = (peg_head_goal_yz_dist < 0.01) & (
            peg_body_goal_yz_dist < 0.01
        )

        hole_radii = env.box_hole_radii.to(self.device, dtype=torch.float32)
        peg_head_hole_x = peg_head_pos_at_hole[:, 0]
        peg_head_hole_abs_y = torch.abs(peg_head_pos_at_hole[:, 1])
        peg_head_hole_abs_z = torch.abs(peg_head_pos_at_hole[:, 2])
        partial_insert_current = (
            prealigned_current
            & (peg_head_hole_x >= -0.05)
            & (peg_head_hole_abs_y <= 1.25 * hole_radii)
            & (peg_head_hole_abs_z <= 1.25 * hole_radii)
        )

        success_current = infos.get("success")
        if success_current is None:
            success_current = (
                (peg_head_hole_x >= -0.015)
                & (peg_head_hole_abs_y <= hole_radii)
                & (peg_head_hole_abs_z <= hole_radii)
            )
        success_current = success_current.to(self.device, dtype=torch.bool)

        consecutive_grasp_event = (
            consecutive_grasp_current & (~self.peg_consecutive_grasp_once)
        )
        prealign_event = prealigned_current & (~self.peg_prealign_once)
        partial_insert_event = (
            partial_insert_current & (~self.peg_partial_insert_once)
        )
        success_event = success_current & (~self.peg_success_once)

        self.peg_consecutive_grasp_once |= consecutive_grasp_current
        self.peg_prealign_once |= prealigned_current
        self.peg_partial_insert_once |= partial_insert_current
        self.peg_success_once |= success_current

        infos.update(
            {
                "peg_head_pos_at_hole": peg_head_pos_at_hole,
                "peg_head_hole_x": peg_head_hole_x,
                "peg_head_hole_abs_y": peg_head_hole_abs_y,
                "peg_head_hole_abs_z": peg_head_hole_abs_z,
                "peg_head_goal_yz_dist": peg_head_goal_yz_dist,
                "peg_body_goal_yz_dist": peg_body_goal_yz_dist,
                "is_grasped_current": is_grasped_current,
                "consecutive_grasp_current": consecutive_grasp_current,
                "prealigned_current": prealigned_current,
                "partial_insert_current": partial_insert_current,
                "success_current": success_current,
                "consecutive_grasp_event": consecutive_grasp_event,
                "prealign_event": prealign_event,
                "partial_insert_event": partial_insert_event,
                "success_event": success_event,
                "consecutive_grasp_once": self.peg_consecutive_grasp_once.clone(),
                "prealign_once": self.peg_prealign_once.clone(),
                "partial_insert_once": self.peg_partial_insert_once.clone(),
                "success_once": self.peg_success_once.clone(),
                "success": success_current,
            }
        )
        return infos

    def _calc_step_reward(self, reward, info):
        if getattr(self.cfg, "reward_mode", "default") == "raw":
            pass
        elif getattr(self.cfg, "reward_mode", "default") == "only_success":
            reward = info["success"] * 1.0
        elif getattr(self.cfg, "reward_mode", "default") == "rlt_events":
            reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
                self.env.unwrapped.device
            )
            reward += info["consecutive_grasp_event"].float() * 0.1
            reward += info["prealign_event"].float() * 0.2
            reward += info["partial_insert_event"].float() * 0.3
            reward += info["success_event"].float() * 1.0
        else:
            reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
                self.env.unwrapped.device
            )  # [B, ]
            reward += info["is_src_obj_grasped"] * 0.1
            reward += info["consecutive_grasp"] * 0.1
            reward += (info["success"] & info["is_src_obj_grasped"]) * 1.0
        # diff
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        if options is None:
            seed = self.seed
            options = (
                {"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {}
            )
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        self._show_goal_site_visual()
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_peg_insertion_event_state(env_idx)
            self._reset_metrics(env_idx)
        else:
            self._reset_peg_insertion_event_state()
            self._reset_metrics()
        return extracted_obs, infos

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        infos = self._augment_peg_insertion_info(infos)
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        step_reward = self._calc_step_reward(_reward, infos)

        infos = self._record_metrics(step_reward, infos)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if isinstance(truncations, bool):
            truncations = torch.tensor([truncations], device=self.device)
            truncations = truncations.repeat(self.num_envs)
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)
        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(infos)
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset(options=options)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def run(self):
        obs, info = self.reset()
        for step in range(100):
            action = self.env.action_space.sample()
            obs, rew, terminations, truncations, infos = self.step(action)
            print(
                f"Step {step}: obs={obs.keys()}, rew={rew.mean()}, terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )

    # render utils
    def capture_image(self, infos=None):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]

        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=int(np.sqrt(self.num_envs)))
        return img

    def render(self, info, rew=None):
        if self.video_cfg.info_on_video:
            scalar_info = gym_utils.extract_scalars_from_info(
                common.to_numpy(info), batch_size=self.num_envs
            )
            if rew is not None:
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
            image = self.capture_image(scalar_info)
        else:
            image = self.capture_image()
        return image

    def sample_action_space(self):
        return self.env.action_space.sample()
