"""Replay buffer for native RLT stage2 TD3 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class TransitionBatch:
    x: torch.Tensor
    a: torch.Tensor
    a_tilde: torch.Tensor
    rewards: torch.Tensor
    next_x: torch.Tensor
    next_a_tilde: torch.Tensor
    dones: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "x": self.x,
            "a": self.a,
            "a_tilde": self.a_tilde,
            "rewards": self.rewards,
            "next_x": self.next_x,
            "next_a_tilde": self.next_a_tilde,
            "dones": self.dones,
        }


class RLTStage2ReplayBuffer:
    """Fixed-capacity circular replay buffer storing chunk-level TD3 transitions."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_chunk_dim: int,
        chunk_length: int,
        seed: int = 1234,
    ) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_chunk_dim = int(action_chunk_dim)
        self.chunk_length = int(chunk_length)

        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

        self._x = np.zeros((capacity, state_dim), dtype=np.float32)
        self._a = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._a_tilde = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, chunk_length), dtype=np.float32)
        self._next_x = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_a_tilde = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def is_ready(self, min_size: int) -> bool:
        return self._size >= int(min_size)

    def add(
        self,
        *,
        x: np.ndarray,
        a: np.ndarray,
        a_tilde: np.ndarray,
        rewards: np.ndarray,
        next_x: np.ndarray,
        next_a_tilde: np.ndarray,
        done: float,
    ) -> None:
        self._x[self._ptr] = x
        self._a[self._ptr] = a
        self._a_tilde[self._ptr] = a_tilde
        self._rewards[self._ptr] = rewards
        self._next_x[self._ptr] = next_x
        self._next_a_tilde[self._ptr] = next_a_tilde
        self._dones[self._ptr] = done

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str) -> TransitionBatch:
        indices = self._rng.integers(0, self._size, size=int(batch_size))
        return TransitionBatch(
            x=torch.as_tensor(self._x[indices], device=device),
            a=torch.as_tensor(self._a[indices], device=device),
            a_tilde=torch.as_tensor(self._a_tilde[indices], device=device),
            rewards=torch.as_tensor(self._rewards[indices], device=device),
            next_x=torch.as_tensor(self._next_x[indices], device=device),
            next_a_tilde=torch.as_tensor(self._next_a_tilde[indices], device=device),
            dones=torch.as_tensor(self._dones[indices], device=device),
        )

    def state_dict(self) -> dict[str, Any]:
        n = self._size
        return {
            "ptr": self._ptr,
            "size": self._size,
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_chunk_dim": self.action_chunk_dim,
            "chunk_length": self.chunk_length,
            "x": self._x[:n].copy(),
            "a": self._a[:n].copy(),
            "a_tilde": self._a_tilde[:n].copy(),
            "rewards": self._rewards[:n].copy(),
            "next_x": self._next_x[:n].copy(),
            "next_a_tilde": self._next_a_tilde[:n].copy(),
            "dones": self._dones[:n].copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        n = int(state["size"])
        self._ptr = int(state["ptr"])
        self._size = n
        self._x[:n] = state["x"]
        self._a[:n] = state["a"]
        self._a_tilde[:n] = state["a_tilde"]
        self._rewards[:n] = state["rewards"]
        self._next_x[:n] = state["next_x"]
        self._next_a_tilde[:n] = state["next_a_tilde"]
        self._dones[:n] = state["dones"]
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state

    def get_stats(self) -> dict[str, float]:
        return {
            "size": float(self._size),
            "capacity": float(self.capacity),
            "fill_ratio": float(self._size / max(self.capacity, 1)),
        }
