"""RLT Stage 2 policy integrated into RLinf.

This policy keeps the original Stage 2 structure:
- frozen OpenPI VLA
- frozen RL token encoder
- trainable residual actor
- trainable twin-Q critic

The policy exposes RLinf-compatible interfaces so the existing rollout/env
pipeline can be reused. Training itself is handled by a dedicated actor worker.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .components import ResidualActor, TwinQCritic, compute_td_target
from .rl_token import RLTokenModel
from .vla_wrapper import Stage2VLAWrapper


class RLTStage2Policy(torch.nn.Module, BasePolicy):
    def __init__(
        self,
        cfg: DictConfig,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        stage2_cfg = cfg.rlt_stage2
        self.chunk_length = int(cfg.num_action_chunks)
        self.action_dim = int(cfg.action_dim)
        self.action_chunk_dim = self.chunk_length * self.action_dim
        self.proprio_dim = int(stage2_cfg.get("proprio_dim", self.action_dim))

        self.vla = Stage2VLAWrapper(
            model_path=cfg.model_path,
            config_name=stage2_cfg.config_name,
            norm_stats_path=stage2_cfg.get("norm_stats_path", None),
            num_images_in_input=int(stage2_cfg.get("num_images_in_input", 1)),
            device=self.device,
        )

        self.rl_token_model = RLTokenModel(
            embedding_dim=int(stage2_cfg.get("embedding_dim", 2048)),
            encoder_layers=int(stage2_cfg.get("encoder_layers", 2)),
            encoder_heads=int(stage2_cfg.get("encoder_heads", 8)),
            decoder_layers=int(stage2_cfg.get("decoder_layers", 2)),
            decoder_heads=int(stage2_cfg.get("decoder_heads", 8)),
        ).to(self.device)
        rl_token_ckpt = torch.load(stage2_cfg.rl_token_path, map_location="cpu")
        if "model_state_dict" in rl_token_ckpt:
            rl_token_ckpt = rl_token_ckpt["model_state_dict"]
        self.rl_token_model.load_state_dict(rl_token_ckpt, strict=False)
        self.rl_token_model.eval()
        for param in self.rl_token_model.parameters():
            param.requires_grad_(False)

        embedding_dim = int(stage2_cfg.get("embedding_dim", 2048))
        self.state_dim = embedding_dim + self.proprio_dim

        self.actor = ResidualActor(
            state_dim=self.state_dim,
            action_chunk_dim=self.action_chunk_dim,
            hidden_dim=int(stage2_cfg.get("mlp_hidden_dim", 256)),
            num_hidden_layers=int(stage2_cfg.get("mlp_num_hidden_layers", 2)),
            sigma=float(stage2_cfg.get("actor_noise_sigma", 0.1)),
            ref_dropout=float(stage2_cfg.get("ref_action_dropout", 0.0)),
        ).to(self.device)

        self.critic = TwinQCritic(
            state_dim=self.state_dim,
            action_chunk_dim=self.action_chunk_dim,
            hidden_dim=int(stage2_cfg.get("mlp_hidden_dim", 256)),
            num_hidden_layers=int(stage2_cfg.get("mlp_num_hidden_layers", 2)),
        ).to(self.device)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Unsupported forward_type for RLT Stage 2: {forward_type}")

    def _encode_state_and_reference(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, a_tilde_flat, _ = self._prepare_features(env_obs)
        return x, a_tilde_flat

    def _prepare_features(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        observation, processed_obs = self.vla.prepare_obs(env_obs)
        embeddings, pad_mask = self.vla.extract_embeddings(observation)
        z_rl = self.rl_token_model.encode(embeddings, pad_mask)
        a_tilde = self.vla.get_rl_chunk_reference(observation, self.chunk_length)
        a_tilde_flat = a_tilde.reshape(a_tilde.shape[0], -1)
        state = observation.state[:, : self.proprio_dim].to(
            device=self.device,
            dtype=torch.float32,
        )
        x = torch.cat([z_rl.to(torch.float32), state], dim=-1)
        return x, a_tilde_flat, processed_obs

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "RLT Stage 2 does not use RLinf PPO-style default_forward."
        )

    def actor_forward(
        self,
        x: torch.Tensor,
        a_tilde: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        return self.actor(x, a_tilde, deterministic=deterministic)

    def critic_forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic(x, actions)

    def critic_min(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.critic.q_min(x, actions)

    @torch.no_grad()
    def compute_td_target_batch(
        self,
        *,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_x: torch.Tensor,
        next_a_tilde: torch.Tensor,
    ) -> torch.Tensor:
        stage2_cfg = self.cfg.rlt_stage2
        return compute_td_target(
            rewards=rewards,
            dones=dones,
            next_x=next_x,
            next_a_tilde=next_a_tilde,
            actor=self.actor,
            critic=self.critic,
            gamma=float(stage2_cfg.get("gamma", self.cfg.get("gamma", 0.99))),
            chunk_length=self.chunk_length,
            target_noise_sigma=float(stage2_cfg.get("target_noise_sigma", 0.2)),
            target_noise_clip=float(stage2_cfg.get("target_noise_clip", 0.5)),
        )

    @torch.no_grad()
    def update_target_networks(self, tau: float) -> None:
        self.critic.update_targets(tau)

    @torch.no_grad()
    def encode_obs(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_state_and_reference(env_obs)

    def sac_forward(
        self,
        obs=None,
        data=None,
        deterministic: bool = False,
        **kwargs,
    ):
        if obs is None:
            obs = data if data is not None else kwargs.get("obs")
        x, a_tilde = self._encode_state_and_reference(obs)
        action = self.actor(x, a_tilde, deterministic=deterministic)
        logprob = torch.zeros(
            action.shape[0], 1, device=action.device, dtype=torch.float32
        )
        return action, logprob, {"x": x, "a_tilde": a_tilde}

    def sac_q_forward(
        self,
        obs=None,
        data=None,
        actions=None,
        state_info=None,
        **kwargs,
    ):
        if state_info is None:
            if obs is None:
                obs = data if data is not None else kwargs.get("obs")
            x, _ = self._encode_state_and_reference(obs)
        else:
            x = state_info["x"]
        q1, q2 = self.critic(x, actions)
        return torch.cat([q1, q2], dim=-1)

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x, a_tilde, processed_obs = self._prepare_features(env_obs)
        deterministic = mode == "eval"
        if deterministic:
            self.actor.eval()
        action_flat = self.actor(x, a_tilde, deterministic=deterministic)
        actions = action_flat.reshape(action_flat.shape[0], self.chunk_length, self.action_dim)
        zeros = torch.zeros(
            action_flat.shape[0], 1, device=action_flat.device, dtype=torch.float32
        )
        result = {
            "prev_logprobs": zeros,
            "prev_values": zeros,
            "forward_inputs": {
                "action": action_flat.detach(),
                "x": x.detach(),
                "a_tilde": a_tilde.detach(),
                "tokenized_prompt": processed_obs["tokenized_prompt"].detach(),
                "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"].detach(),
            },
        }
        return actions, result
