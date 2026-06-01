# Copyright 2026 The RLinf Authors.
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
import logging
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils._pytree import tree_map
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

logger = logging.getLogger(__name__)


def _create_local_lerobot_dataset(
    repo_path: str,
    *,
    data_config: Any,
    action_horizon: int,
):
    """Create a local LeRobot dataset without any Hugging Face Hub calls."""
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import openpi.training.data_loader as openpi_data_loader
    import openpi.transforms as transforms

    local_path = Path(repo_path).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Local LeRobot dataset path does not exist: {local_path}")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
        local_path.name,
        root=local_path,
    )
    dataset = lerobot_dataset.LeRobotDataset(
        local_path.name,
        root=local_path,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = openpi_data_loader.TransformedDataset(
            dataset,
            [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )

    return dataset


def _resolve_local_lerobot_repo_path(
    data_kwargs: Any,
    data_paths: list[Any],
) -> str | None:
    """Return the absolute local dataset path if the caller configured one."""
    repo_id = None
    if data_kwargs is not None:
        repo_id = getattr(data_kwargs, "get", lambda *_: None)("repo_id")
        if repo_id is None:
            repo_id = getattr(data_kwargs, "repo_id", None)
    if isinstance(repo_id, str) and os.path.isabs(repo_id):
        return repo_id

    if len(data_paths) == 1:
        first_path = data_paths[0]
        if isinstance(first_path, str) and os.path.isabs(first_path):
            return first_path

        dataset_path = None
        if isinstance(first_path, dict):
            dataset_path = first_path.get("dataset_path")
        else:
            dataset_path = getattr(first_path, "dataset_path", None)
            if dataset_path is None and hasattr(first_path, "get"):
                dataset_path = first_path.get("dataset_path")

        if isinstance(dataset_path, str) and os.path.isabs(dataset_path):
            return dataset_path

    return None


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.RLT_STAGE1,
        ]:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.RLT_STAGE1:
                config_name = self.cfg.actor.model.rlt_stage1.config_name
                batch_size = self.cfg.actor.micro_batch_size * self._world_size
                data_kwargs = getattr(self.cfg.actor, "openpi_data", None)
            else:
                config_name = self.cfg.actor.model.openpi.config_name
                batch_size = self.cfg.actor.micro_batch_size * self._world_size
                data_kwargs = getattr(self.cfg.actor, "openpi_data", None)

            config = get_openpi_config(
                config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=batch_size,
                data_kwargs=data_kwargs,
            )
            data_config = config.data.create(config.assets_dirs, config.model)
            local_repo_path = _resolve_local_lerobot_repo_path(data_kwargs, data_paths)
            logger.info(
                "OpenPI SFT/Stage1 dataloader repo resolution: config_name=%s repo_id=%s local_repo_path=%s",
                config_name,
                getattr(data_config, "repo_id", None),
                local_repo_path,
            )

            if local_repo_path is not None:
                logger.info(
                    "Using local LeRobot dataset path for OpenPI SFT/Stage1 dataloader: %s",
                    local_repo_path,
                )
                dataset = _create_local_lerobot_dataset(
                    local_repo_path,
                    data_config=data_config,
                    action_horizon=config.model.action_horizon,
                )
                dataset = openpi_data_loader.transform_dataset(
                    dataset,
                    data_config,
                    skip_norm_stats=False,
                )

                sampler = None
                if torch.distributed.is_initialized():
                    sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset,
                        num_replicas=torch.distributed.get_world_size(),
                        rank=torch.distributed.get_rank(),
                        shuffle=True,
                        drop_last=True,
                    )
                    local_batch_size = (
                        batch_size // torch.distributed.get_world_size()
                    )
                else:
                    local_batch_size = batch_size

                torch_loader = openpi_data_loader.TorchDataLoader(
                    dataset,
                    local_batch_size=local_batch_size,
                    sharding=None,
                    shuffle=(sampler is None),
                    sampler=sampler,
                    num_batches=None,
                    num_workers=config.num_workers,
                    seed=config.seed,
                    framework="pytorch",
                )
                data_loader = openpi_data_loader.DataLoaderImpl(
                    data_config, torch_loader
                )
            else:
                data_loader = openpi_data_loader.create_data_loader(
                    config, framework="pytorch", shuffle=True
                )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO
        ]:
            self._dreamzero_loss = None
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: dict[str, Any]):
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA,
            SupportedModel.DREAMZERO,
        ]:
            with self.amp_context:
                losses_dict = self.model(forward_type=ForwardType.SFT, data=batch)
            if losses_dict.get("dynamics_loss", None) is not None:
                self._dreamzero_loss = {
                    "dynamics_loss": losses_dict["dynamics_loss"],
                    "action_loss": losses_dict["action_loss"],
                }
            return losses_dict["loss"]
        observation, actions = batch

        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: (
                torch.as_tensor(x, device=self.device).contiguous().clone()
                if x is not None
                else x
            ),
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        with self.amp_context:
            losses = self.model(
                forward_type=ForwardType.SFT,
                data={"observation": observation, "actions": actions},
            )

        # train model return the loss
        return losses

    def run_training(self):
        train_metrics = super().run_training()
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            in [SupportedModel.DREAMZERO]
            and self._dreamzero_loss is not None
        ):
            train_metrics.update(
                {
                    "dynamics_loss": self._dreamzero_loss["dynamics_loss"],
                    "action_loss": self._dreamzero_loss["action_loss"],
                }
            )
            self._dreamzero_loss = None
        return train_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

        rng_state = get_rng_state()
        all_rng_states = [None] * self._world_size
        torch.distributed.all_gather_object(all_rng_states, rng_state)
        if self._rank == 0:
            torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

        torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

        rng_path = os.path.join(load_path, "rng.pt")
        if os.path.exists(rng_path):
            all_rng_states = torch.load(rng_path, weights_only=False)
            set_rng_state(all_rng_states[self._rank])

        torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.RLT_STAGE1,
        ]:
            num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
