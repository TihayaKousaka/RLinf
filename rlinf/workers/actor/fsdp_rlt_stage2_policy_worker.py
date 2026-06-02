"""Native RLinf actor worker for RLT stage2 TD3 training."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory

from ...models.embodiment.rlt_stage2.components import actor_loss, critic_loss
from ...models.embodiment.rlt_stage2.replay_buffer import RLTStage2ReplayBuffer


class RLTStage2FSDPPolicyWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.version = 0

        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

        self.replay_buffer: RLTStage2ReplayBuffer | None = None
        self.qf_optimizer = None
        self.qf_lr_scheduler = None
        self.update_step = 0
        self.gradient_accumulation = 1
        self.actor_only_train_model = bool(
            cfg.algorithm.get("actor_only_train_model", True)
        )
        self._rollout_sync_key_count = 0

        weight_syncer_cfg = cfg.get("weight_syncer", None)
        assert weight_syncer_cfg is not None, (
            "weight_syncer config must be provided for RLT stage2 actor worker."
        )
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

    def _resolve_actor_loss_weights(self) -> tuple[float, float, float, bool, float]:
        stage2_cfg = self.cfg.actor.model.rlt_stage2
        warmup_updates = int(
            self.cfg.algorithm.get(
                "actor_warmup_updates",
                self.cfg.algorithm.get("actor_warmup_steps", 0),
            )
        )
        in_warmup = self.update_step < warmup_updates
        warmup_bc_weight = float(
            stage2_cfg.get(
                "warmup_bc_weight",
                stage2_cfg.get("bc_regularizer_beta", 1.0),
            )
        )
        warmup_q_weight = float(stage2_cfg.get("warmup_q_weight", 0.1))
        online_bc_weight = float(
            stage2_cfg.get(
                "online_bc_weight",
                stage2_cfg.get("bc_regularizer_beta", 1.0),
            )
        )
        online_q_weight = float(stage2_cfg.get("online_q_weight", 0.1))
        if in_warmup:
            bc_weight = warmup_bc_weight
            q_weight = warmup_q_weight
            ramp_progress = 0.0
        else:
            ramp_updates = int(self.cfg.algorithm.get("actor_loss_ramp_updates", 0))
            if ramp_updates > 0:
                ramp_progress = min(
                    1.0,
                    max(
                        0.0,
                        float(self.update_step - warmup_updates + 1)
                        / float(ramp_updates),
                    ),
                )
            else:
                ramp_progress = 1.0
            bc_weight = warmup_bc_weight + ramp_progress * (
                online_bc_weight - warmup_bc_weight
            )
            q_weight = warmup_q_weight + ramp_progress * (
                online_q_weight - warmup_q_weight
            )
        delta_weight = float(stage2_cfg.get("delta_weight", 0.0))
        return bc_weight, q_weight, delta_weight, in_warmup, ramp_progress

    def init_worker(self):
        self.setup_model_and_optimizer()
        self._init_replay_buffer()
        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            dst_rank = i * actor_world_size + rank
            if dst_rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(dst_rank)

    def model_provider_func(self) -> torch.nn.Module:
        from rlinf.models import get_model

        model_cfg = self.cfg.actor.model
        if bool(
            self.actor_only_train_model
        ) and model_cfg.get("rlt_stage2", None) is not None:
            from copy import deepcopy
            from omegaconf import open_dict

            model_cfg = deepcopy(model_cfg)
            with open_dict(model_cfg):
                model_cfg.rlt_stage2.load_feature_backbones = False
                model_cfg.rlt_stage2.load_rl_token_model = False

        model = get_model(model_cfg)
        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path, map_location="cpu")
            model.load_state_dict(model_dict)
        return model

    def setup_model_and_optimizer(self) -> None:
        module = self.model_provider_func()
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype

        actor_filters = {"critic": ["critic."]}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=actor_filters,
            filtered_optim_config={"critic": self.cfg.actor.critic_optim},
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        grad_scaler_cfg = self.cfg.actor.fsdp_config.grad_scaler
        kwargs = {}
        for key in ["init_scale", "growth_interval"]:
            value = grad_scaler_cfg.get(key, None)
            if value is not None:
                kwargs[key] = value
        self.grad_scaler = self.build_grad_scaler(
            grad_scaler_cfg.get("enabled", False),
            **kwargs,
        )

    def _init_replay_buffer(self) -> None:
        stage2_cfg = self.cfg.actor.model.rlt_stage2
        capacity = int(stage2_cfg.get("buffer_capacity", 200000))
        self.replay_buffer = RLTStage2ReplayBuffer(
            capacity=capacity,
            state_dim=int(self.cfg.actor.model.rlt_stage2.embedding_dim)
            + int(self.cfg.actor.model.rlt_stage2.proprio_dim),
            action_chunk_dim=int(self.cfg.actor.model.num_action_chunks)
            * int(self.cfg.actor.model.action_dim),
            chunk_length=int(self.cfg.actor.model.num_action_chunks),
            seed=int(self.cfg.actor.get("seed", 1234)) + self._rank,
        )

    def get_rollout_state_dict(self) -> dict:
        state_dict = self.model.filter_rollout_state_dict(
            self.get_model_state_dict(cpu_offload=False, full_state_dict=False)
        )
        self._rollout_sync_key_count = len(state_dict)
        return state_dict

    async def sync_model_to_rollout(self) -> None:
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            handles = []
            for rank in self._weight_dst_rank_in_rollout:
                handles.append(
                    self.send(
                        data,
                        dst_group_name=self._rollout_group_name,
                        dst_rank=rank,
                        async_op=True,
                        options=self._sync_weight_comm_options,
                    ).async_wait()
                )
            await asyncio.gather(*handles)

        async def recv_func():
            handles = []
            for rank in self._weight_dst_rank_in_rollout:
                handles.append(
                    self.recv(
                        src_group_name=self._rollout_group_name,
                        src_rank=rank,
                        async_op=True,
                        options=self._sync_weight_comm_options,
                    ).async_wait()
                )
            metadata_list = await asyncio.gather(*handles)
            return metadata_list[0]

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
            )

        await self.weight_syncer.sync(state_dict, send_func, version=self.version)

        if self.enable_offload:
            self.offload_param_and_grad(True)

    def _trajectory_to_transitions(self, traj: Trajectory) -> int:
        if self.replay_buffer is None or traj.actions is None or not traj.forward_inputs:
            return 0

        traj_len = traj.actions.shape[0]
        bsz = traj.actions.shape[1]
        added = 0

        x_all = traj.forward_inputs.get("x")
        a_tilde_all = traj.forward_inputs.get("a_tilde")
        if x_all is None or a_tilde_all is None:
            return 0

        dones_all = traj.dones
        rewards_all = traj.rewards
        if dones_all is None or rewards_all is None:
            return 0
        intervention_flags_all = traj.forward_inputs.get("intervention_flags")
        if intervention_flags_all is None:
            intervention_flags_all = traj.intervene_flags

        for env_idx in range(bsz):
            for t in range(traj_len):
                done_idx = min(t + 1, dones_all.shape[0] - 1)
                done = float(dones_all[done_idx, env_idx].any().item())
                intervention = 0.0
                if intervention_flags_all is not None:
                    intervention = float(
                        intervention_flags_all[t, env_idx].detach().float().mean().item()
                    )

                x = x_all[t, env_idx].detach().cpu().numpy().astype(np.float32, copy=False)
                a_tilde = (
                    a_tilde_all[t, env_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                action = (
                    traj.actions[t, env_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                rewards = (
                    rewards_all[t, env_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                if done > 0.0:
                    next_x = x
                    next_a_tilde = a_tilde
                elif t + 1 < traj_len:
                    next_x = (
                        x_all[t + 1, env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    next_a_tilde = (
                        a_tilde_all[t + 1, env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                else:
                    if x_all.shape[0] <= t + 1 or a_tilde_all.shape[0] <= t + 1:
                        raise RuntimeError(
                            "RLT Stage2 rollout boundary transition is non-terminal "
                            "but missing cached final x/a_tilde. Rollout must send "
                            "the final student forward_inputs so actor training can "
                            "bootstrap without re-encoding VLA observations."
                        )
                    next_x = (
                        x_all[t + 1, env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    next_a_tilde = (
                        a_tilde_all[t + 1, env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )

                self.replay_buffer.add(
                    x=x,
                    a=action,
                    a_tilde=a_tilde,
                    rewards=rewards,
                    next_x=next_x,
                    next_a_tilde=next_a_tilde,
                    done=done,
                    intervention=intervention,
                )
                added += 1
                if done > 0.0:
                    break

        return added

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            self._trajectory_to_transitions(trajectory)

    @Worker.timer("run_training")
    def run_training(self):
        if self.replay_buffer is None:
            return {}

        if self.enable_offload:
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 1)
        if not self.replay_buffer.is_ready(min_buffer_size):
            return {}

        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        assert global_batch_size_per_rank % self.cfg.actor.micro_batch_size == 0, (
            "global batch per rank must be divisible by micro_batch_size"
        )
        micro_batch_cnt = global_batch_size_per_rank // self.cfg.actor.micro_batch_size
        self.gradient_accumulation = micro_batch_cnt

        self.model.train()
        metrics = {}
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        for _ in range(update_epoch):
            batch = self.replay_buffer.sample(global_batch_size_per_rank, self.device)
            batch_dict = batch.to_dict()
            micro_batches = []
            for i in range(micro_batch_cnt):
                begin = i * self.cfg.actor.micro_batch_size
                end = begin + self.cfg.actor.micro_batch_size
                micro_batches.append({k: v[begin:end] for k, v in batch_dict.items()})
            epoch_metrics = self._update_one_epoch(micro_batches)
            append_to_dict(metrics, epoch_metrics)
            self.update_step += 1

        stats = self.replay_buffer.get_stats()
        for key, value in stats.items():
            append_to_dict(metrics, {f"replay_buffer/{key}": value})
        append_to_dict(
            metrics,
            {
                "rlt_stage2/actor_only_train_model": float(
                    self.actor_only_train_model
                ),
                "rlt_stage2/rollout_sync_key_count": float(
                    self._rollout_sync_key_count
                ),
            },
        )
        return self._process_train_metrics(metrics)

    def _update_one_epoch(self, micro_batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        stage2_cfg = self.cfg.actor.model.rlt_stage2
        critic_losses = []
        q1_values = []
        q2_values = []

        self.optimizer.zero_grad(set_to_none=True)
        self.qf_optimizer.zero_grad(set_to_none=True)
        for idx, batch in enumerate(micro_batches):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == len(micro_batches),
            )
            with backward_ctx:
                with self.amp_context:
                    td_target = self.model.compute_td_target_batch(
                        rewards=batch["rewards"].to(torch.float32),
                        dones=batch["dones"].to(torch.float32),
                        next_x=batch["next_x"].to(torch.float32),
                        next_a_tilde=batch["next_a_tilde"].to(torch.float32),
                    )
                    q1, q2 = self.model.critic_forward(
                        batch["x"].to(torch.float32),
                        batch["a"].to(torch.float32),
                    )
                    loss = (
                        critic_loss(q1, q2, td_target) / self.gradient_accumulation
                    )
                self.grad_scaler.scale(loss).backward()
            critic_losses.append(loss.detach().float().item() * self.gradient_accumulation)
            q1_values.append(q1.detach().float().mean().item())
            q2_values.append(q2.detach().float().mean().item())

        self.grad_scaler.unscale_(self.qf_optimizer)
        critic_grad_norm = self._strategy.clip_grad_norm_(self.model)
        self.grad_scaler.step(self.qf_optimizer)
        self.grad_scaler.update()
        self.qf_lr_scheduler.step()
        self.qf_optimizer.zero_grad(set_to_none=True)

        metrics = {
            "rlt_stage2/critic_loss": float(np.mean(critic_losses)),
            "critic/q1_mean": float(np.mean(q1_values)),
            "critic/q2_mean": float(np.mean(q2_values)),
            "critic/grad_norm": float(critic_grad_norm),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
        }

        min_buffer_size = int(
            self.cfg.algorithm.replay_buffer.get("min_buffer_size", 1)
        )
        actor_warmup_steps = int(
            self.cfg.algorithm.get("actor_warmup_steps", min_buffer_size)
        )
        actor_warmup_done = len(self.replay_buffer) >= actor_warmup_steps
        bc_weight, q_weight, delta_weight, in_loss_warmup, loss_ramp_progress = (
            self._resolve_actor_loss_weights()
        )
        update_actor = (
            actor_warmup_done
            and (self.update_step + 1) % int(self.cfg.algorithm.critic_actor_ratio) == 0
        )
        if update_actor:
            actor_losses = []
            actor_q_values = []
            actor_residual_abs = []
            actor_residual_l2 = []
            actor_bc_losses = []
            actor_delta_losses = []
            self.optimizer.zero_grad()
            self.model.set_online_critic_requires_grad(False)
            for idx, batch in enumerate(micro_batches):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == len(micro_batches),
                )
                with backward_ctx:
                    with self.amp_context:
                        actions = self.model.actor_forward(
                            batch["x"].to(torch.float32),
                            batch["a_tilde"].to(torch.float32),
                            deterministic=True,
                            apply_ref_dropout=bool(
                                stage2_cfg.get("ref_action_dropout", 0.0) > 0.0
                            ),
                            apply_action_noise=False,
                        )
                        a_tilde_flat = batch["a_tilde"].to(torch.float32)
                        residual = actions - a_tilde_flat
                        chunk_len = int(self.cfg.actor.model.num_action_chunks)
                        action_dim = int(self.cfg.actor.model.action_dim)
                        actions_chunk = actions.reshape(-1, chunk_len, action_dim)
                        a_tilde_chunk = a_tilde_flat.reshape(-1, chunk_len, action_dim)
                        q_value = self.model.critic_min(
                            batch["x"].to(torch.float32),
                            actions,
                        )
                        actor_total_loss, actor_loss_metrics = actor_loss(
                            q_value=q_value,
                            a=actions_chunk,
                            a_tilde=a_tilde_chunk,
                            bc_weight=bc_weight,
                            q_weight=q_weight,
                            delta_weight=delta_weight,
                        )
                        loss = actor_total_loss / self.gradient_accumulation
                    self.grad_scaler.scale(loss).backward()
                actor_losses.append(loss.detach().float().item() * self.gradient_accumulation)
                actor_q_values.append(q_value.detach().float().mean().item())
                actor_residual_abs.append(residual.detach().float().abs().mean().item())
                actor_residual_l2.append(
                    residual.detach().float().pow(2).mean().sqrt().item()
                )
                actor_bc_losses.append(
                    float(actor_loss_metrics["bc_loss"].float().item())
                )
                actor_delta_losses.append(
                    float(actor_loss_metrics["delta_loss"].float().item())
                )
            self.model.set_online_critic_requires_grad(True)

            self.grad_scaler.unscale_(self.optimizer)
            actor_grad_norm = self._strategy.clip_grad_norm_(self.model)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            metrics.update(
                {
                    "rlt_stage2/actor_loss": float(np.mean(actor_losses)),
                    "actor/q_mean": float(np.mean(actor_q_values)),
                    "actor/grad_norm": float(actor_grad_norm),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/residual_abs_mean": float(np.mean(actor_residual_abs)),
                    "actor/residual_l2": float(np.mean(actor_residual_l2)),
                    "actor/bc_weight": bc_weight,
                    "actor/q_weight": q_weight,
                    "actor/delta_weight": delta_weight,
                    "actor/bc_loss": float(np.mean(actor_bc_losses)),
                    "actor/delta_loss": float(np.mean(actor_delta_losses)),
                    "actor/loss_warmup": float(in_loss_warmup),
                    "actor/loss_ramp_progress": loss_ramp_progress,
                }
            )
        else:
            metrics.update(
                {
                    "actor/update_skipped": 1.0,
                    "actor/warmup_done": float(actor_warmup_done),
                    "actor/warmup_steps": float(actor_warmup_steps),
                    "actor/bc_weight": bc_weight,
                    "actor/q_weight": q_weight,
                    "actor/delta_weight": delta_weight,
                    "actor/loss_warmup": float(in_loss_warmup),
                    "actor/loss_ramp_progress": loss_ramp_progress,
                }
            )

        self.model.update_target_networks(float(self.cfg.algorithm.tau))
        return metrics

    def _process_train_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                mean_metric_dict[key] = float(np.mean(value))
            elif isinstance(value, torch.Tensor):
                mean_metric_dict[key] = float(value.detach().cpu().item())
            else:
                mean_metric_dict[key] = float(value)
        return all_reduce_dict(mean_metric_dict, op=torch.distributed.ReduceOp.AVG)

    def compute_advantages_and_returns(self):
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        stage2_save_path = os.path.join(save_base_path, "rlt_stage2_components")
        os.makedirs(stage2_save_path, exist_ok=True)
        torch.save(
            {
                "update_step": self.update_step,
                "version": self.version,
                "replay_buffer": self.replay_buffer.state_dict()
                if self.replay_buffer is not None
                else None,
            },
            os.path.join(stage2_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

    def load_checkpoint(self, load_base_path):
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        stage2_load_path = os.path.join(
            load_base_path,
            "rlt_stage2_components",
            f"checkpoint_rank_{self._rank}.pt",
        )
        if os.path.exists(stage2_load_path):
            state = torch.load(stage2_load_path, map_location="cpu", weights_only=False)
            self.update_step = int(state.get("update_step", 0))
            self.version = int(state.get("version", self.update_step))
            if self.replay_buffer is not None and state.get("replay_buffer") is not None:
                self.replay_buffer.load_state_dict(state["replay_buffer"])

    def set_global_step(self, global_step: int) -> None:
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
