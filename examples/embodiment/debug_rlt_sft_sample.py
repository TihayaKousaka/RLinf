#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

    for candidate in (rlinf_root, repo_root / "openpi-RLT" / "src"):
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
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

from rlinf.config import validate_cfg
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.utils.pytree import register_pytree_dataclasses

from openpi.models import model as _openpi_model


_INDEX_OVERRIDE_RE = re.compile(r"^(?P<prefix>[\w.]+)\[(?P<index>\d+)\]\.(?P<suffix>[^=]+)=(?P<value>.*)$")


def _print_section(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def _to_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach()
    try:
        return torch.as_tensor(value)
    except Exception:
        return None


def _stats(name: str, value: Any) -> None:
    tensor = _to_tensor(value)
    if tensor is None:
        print(f"{name}: <non-numeric {type(value).__name__}>")
        return

    tensor = tensor.to(torch.float32)
    shape = tuple(tensor.shape)
    flat = tensor.reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite_vals = flat[finite_mask]
    if finite_vals.numel() == 0:
        print(f"{name}: shape={shape} no_finite_values")
        return

    std = finite_vals.std(unbiased=False).item() if finite_vals.numel() > 1 else 0.0
    print(
        f"{name}: shape={shape} "
        f"mean={finite_vals.mean().item():.6f} std={std:.6f} "
        f"min={finite_vals.min().item():.6f} max={finite_vals.max().item():.6f} "
        f"abs_mean={finite_vals.abs().mean().item():.6f}"
    )


def _first_text(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return str(value)


def _summarize_sample(sample: dict[str, Any], prefix: str = "sample") -> None:
    for key, value in sample.items():
        if isinstance(value, (str, bytes)):
            print(f"{prefix}[{key}]: {_first_text(value)[:120]}")
            continue
        if isinstance(value, dict):
            print(f"{prefix}[{key}]: dict")
            _summarize_sample(value, prefix=f"{prefix}[{key}]")
            continue
        tensor = _to_tensor(value)
        if tensor is not None:
            _stats(f"{prefix}[{key}]", value)
        else:
            print(f"{prefix}[{key}]: type={type(value).__name__}")


def _summarize_observation(obs: Any) -> None:
    for key, value in obs.__dict__.items():
        if value is None:
            print(f"observation.{key}: <none>")
            continue
        if isinstance(value, dict):
            print(f"observation.{key}:")
            for sub_key, sub_value in value.items():
                _stats(f"  {sub_key}", sub_value)
        elif isinstance(value, list):
            print(f"observation.{key}: list[{len(value)}]")
            for idx, item in enumerate(value):
                _stats(f"  [{idx}]", item)
        else:
            _stats(f"observation.{key}", value)


def _add_batch_dim(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [value]
    return torch.as_tensor(value).unsqueeze(0)


def _move_observation_to_device(observation: Any, device: torch.device) -> Any:
    register_pytree_dataclasses(observation)
    return tree_map(
        lambda x: (
            torch.as_tensor(x, device=device).contiguous().clone() if x is not None else x
        ),
        observation,
    )


def _prepare_training_batch(transformed_sample: dict[str, Any], device: torch.device):
    batched = tree_map(_add_batch_dim, transformed_sample)
    if "actions" not in batched:
        raise KeyError("Transformed training sample does not contain an 'actions' key.")

    actions = batched.pop("actions").to(torch.float32).to(device)
    observation = _openpi_model.Observation.from_dict(batched)
    observation = _move_observation_to_device(observation, device)
    return observation, actions


def _image_to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(np.asarray(value))


def _prompt_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        try:
            return str(value.item())
        except Exception:
            pass
    return str(value)


def _build_rollout_like_obs(source_sample: dict[str, Any]) -> dict[str, Any]:
    if "image" not in source_sample:
        raise KeyError("Source sample does not contain an 'image' key.")

    image = source_sample["image"]
    wrist_image = source_sample.get("wrist_image")
    extra_view_image = source_sample.get("extra_view_image")
    if wrist_image is None and extra_view_image is not None:
        wrist_image = extra_view_image

    prompt = source_sample.get("prompt")
    if prompt is None:
        prompt = source_sample.get("task")
    if prompt is None:
        prompt = "insert the peg into the hole"

    return {
        "main_images": _image_to_tensor(image).unsqueeze(0),
        "wrist_images": (
            _image_to_tensor(wrist_image).unsqueeze(0) if wrist_image is not None else None
        ),
        "extra_view_images": None,
        "states": torch.as_tensor(source_sample["state"], dtype=torch.float32).unsqueeze(0),
        "task_descriptions": [_prompt_to_string(prompt)],
    }


def _get_raw_gt_actions(source_sample: dict[str, Any]) -> torch.Tensor:
    if "actions" not in source_sample:
        raise KeyError("Source sample does not contain an 'actions' key.")
    actions = torch.as_tensor(np.asarray(source_sample["actions"]), dtype=torch.float32)
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    return actions


def _mean_sft_loss(model, observation: Any, actions: torch.Tensor) -> tuple[torch.Tensor, Any]:
    losses = model(
        forward_type=ForwardType.SFT,
        data={"observation": observation, "actions": actions},
    )
    if isinstance(losses, (list, tuple)):
        loss_tensor = torch.stack(
            [item if torch.is_tensor(item) else torch.as_tensor(item) for item in losses]
        )
    elif torch.is_tensor(losses):
        loss_tensor = losses
    else:
        loss_tensor = torch.as_tensor(losses, device=actions.device, dtype=torch.float32)
    return loss_tensor.mean(), losses


def _predict_actions(model, source_sample: dict[str, Any], seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()
    env_obs = _build_rollout_like_obs(source_sample)
    with torch.no_grad():
        return model.predict_action_batch(
            env_obs,
            mode="eval",
            compute_values=False,
        )


def _print_action_values(name: str, actions: torch.Tensor) -> None:
    actions = actions.detach().cpu().to(torch.float32)
    first = actions[0, 0].tolist()
    print(f"{name}_first_step={[round(value, 6) for value in first]}")


def _compare_model_output_to_ground_truth(
    model_output_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
) -> None:
    model_output_actions = model_output_actions.detach().cpu().to(torch.float32)
    ground_truth_actions = ground_truth_actions.detach().cpu().to(torch.float32)
    compare_horizon = min(model_output_actions.shape[1], ground_truth_actions.shape[1])
    compare_dim = min(model_output_actions.shape[2], ground_truth_actions.shape[2])
    print(
        "action_compare_shape "
        f"model={tuple(model_output_actions.shape)} "
        f"ground_truth={tuple(ground_truth_actions.shape)} "
        f"using_horizon={compare_horizon} using_dim={compare_dim}"
    )
    if model_output_actions.shape[1] != ground_truth_actions.shape[1]:
        print(
            "action_compare_note="
            "model and ground-truth horizons differ; comparing only the shared prefix."
        )
    model_output_actions = model_output_actions[:, :compare_horizon, :compare_dim]
    ground_truth_actions = ground_truth_actions[:, :compare_horizon, :compare_dim]

    _print_action_values("model_output_action", model_output_actions)
    _print_action_values("ground_truth_action", ground_truth_actions)
    _compare_actions(model_output_actions, ground_truth_actions)


def _fit_single_sample(
    model,
    observation: Any,
    actions: torch.Tensor,
    steps: int,
    lr: float,
    log_interval: int,
) -> None:
    if steps <= 0:
        return

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)

    _print_section("Single-Sample Fit")
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _mean_sft_loss(model, observation, actions)
        loss.backward()
        optimizer.step()

        if step == 1 or step == steps or step % log_interval == 0:
            print(f"fit_step={step} loss={loss.detach().cpu().item():.8f}")


def _compare_actions(pred: torch.Tensor, gt: torch.Tensor) -> None:
    pred = pred.detach().cpu().to(torch.float32)
    gt = gt.detach().cpu().to(torch.float32)
    diff = pred - gt
    abs_diff = diff.abs()

    _stats("pred_actions", pred)
    _stats("gt_actions", gt)
    _stats("abs_diff", abs_diff)

    first_pred = pred[:, 0]
    first_gt = gt[:, 0]
    first_abs = (first_pred - first_gt).abs()
    action_dim = pred.shape[-1]
    if action_dim == 7:
        print(
            "first_step_mae xyz="
            f"{first_abs[:, :3].mean().item():.6f} "
            "rpy="
            f"{first_abs[:, 3:6].mean().item():.6f} "
            "gripper="
            f"{first_abs[:, 6:7].mean().item():.6f}"
        )
        print(
            "chunk_mae xyz="
            f"{abs_diff[:, :, :3].mean().item():.6f} "
            "rpy="
            f"{abs_diff[:, :, 3:6].mean().item():.6f} "
            "gripper="
            f"{abs_diff[:, :, 6:7].mean().item():.6f}"
        )
    elif action_dim == 8:
        print(
            "first_step_mae arm="
            f"{first_abs[:, :7].mean().item():.6f} "
            "gripper="
            f"{first_abs[:, 7:8].mean().item():.6f}"
        )
        print(
            "chunk_mae arm="
            f"{abs_diff[:, :, :7].mean().item():.6f} "
            "gripper="
            f"{abs_diff[:, :, 7:8].mean().item():.6f}"
        )
    else:
        print(f"first_step_mae all={first_abs.mean().item():.6f}")
        print(f"chunk_mae all={abs_diff.mean().item():.6f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug one RLT SFT training sample by comparing the training-path loss "
            "and the rollout-path predicted action against ground-truth action."
        )
    )
    parser.add_argument(
        "--config-name",
        default="rlt_maniskill_pi05_sft",
        help="Hydra config name under RLinf/examples/sft/config",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index in the transformed torch dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic debug output.",
    )
    parser.add_argument(
        "--fit-steps",
        type=int,
        default=0,
        help="Optional number of optimizer steps to overfit this one transformed training sample.",
    )
    parser.add_argument(
        "--fit-lr",
        type=float,
        default=1e-5,
        help="Learning rate used by --fit-steps.",
    )
    parser.add_argument(
        "--fit-log-interval",
        type=int,
        default=10,
        help="How often to print single-sample fit loss.",
    )
    parser.add_argument(
        "--keep-compile",
        action="store_true",
        help="Keep the model's torch.compile setting. By default this debug script disables it to start faster.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Checkpoint/model path. Equivalent to actor.model.model_path.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="LeRobot dataset path. Also used as actor.openpi_data.repo_id unless --repo-id is set.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="OpenPI data repo_id/path. Defaults to --dataset-path.",
    )
    parser.add_argument(
        "--norm-stats-path",
        default=None,
        help="Norm stats json/path. Equivalent to actor.openpi_data.norm_stats_path.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Extra Hydra overrides, e.g. "
            "data.train_data_paths[0].dataset_path=/abs/path "
            "actor.openpi_data.repo_id=/abs/path "
            "actor.openpi_data.norm_stats_path=/abs/path/norm_stats.json "
            "actor.model.model_path=/abs/path/to/checkpoint"
        ),
    )
    return parser.parse_args()


def _split_index_overrides(overrides: list[str]) -> tuple[list[str], list[tuple[str, int, str, str]]]:
    hydra_overrides = []
    index_overrides = []
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


def _apply_index_overrides(cfg, index_overrides: list[tuple[str, int, str, str]]) -> None:
    for prefix, index, suffix, value in index_overrides:
        container = OmegaConf.select(cfg, prefix)
        if container is None:
            raise KeyError(f"Override target does not exist: {prefix}[{index}].{suffix}")
        if index >= len(container):
            raise IndexError(
                f"Override target index out of range: {prefix}[{index}].{suffix}"
            )
        OmegaConf.update(container[index], suffix, value, merge=True)


def _apply_direct_path_args(cfg, args: argparse.Namespace) -> None:
    if args.model_path:
        OmegaConf.update(cfg, "actor.model.model_path", args.model_path, merge=True)

    if args.dataset_path:
        train_data_paths = cfg.data.get("train_data_paths")
        if train_data_paths is None or len(train_data_paths) == 0:
            OmegaConf.update(
                cfg,
                "data.train_data_paths",
                [{"dataset_path": args.dataset_path, "weight": 1.0}],
                merge=True,
            )
        else:
            OmegaConf.update(
                cfg.data.train_data_paths[0],
                "dataset_path",
                args.dataset_path,
                merge=True,
            )
        if not args.repo_id:
            OmegaConf.update(
                cfg,
                "actor.openpi_data.repo_id",
                args.dataset_path,
                merge=True,
            )

    if args.repo_id:
        OmegaConf.update(cfg, "actor.openpi_data.repo_id", args.repo_id, merge=True)

    if args.norm_stats_path:
        OmegaConf.update(
            cfg,
            "actor.openpi_data.norm_stats_path",
            args.norm_stats_path,
            merge=True,
        )


def _compose_cfg(args: argparse.Namespace):
    config_dir = RLINF_ROOT / "examples" / "sft" / "config"
    hydra_overrides, index_overrides = _split_index_overrides(list(args.overrides))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name=args.config_name, overrides=hydra_overrides)
    _apply_index_overrides(cfg, index_overrides)
    _apply_direct_path_args(cfg, args)
    cfg = validate_cfg(cfg)
    if not args.keep_compile:
        OmegaConf.update(
            cfg,
            "actor.model.openpi.pytorch_compile_mode",
            None,
            force_add=True,
        )
    return cfg


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = _compose_cfg(args)

    if not os.environ.get("HF_LEROBOT_HOME"):
        train_data_paths = cfg.data.get("train_data_paths", [])
        if len(train_data_paths) > 0:
            first_dataset = train_data_paths[0].get("dataset_path")
            if first_dataset:
                os.environ["HF_LEROBOT_HOME"] = os.path.dirname(first_dataset)

    _print_section("Resolved Config")
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    config = get_openpi_config(
        cfg.actor.model.openpi.config_name,
        model_path=cfg.actor.model.model_path,
        batch_size=1,
        data_kwargs=cfg.actor.openpi_data,
    )

    import openpi.training.data_loader as openpi_data_loader

    data_config = config.data.create(config.assets_dirs, config.model)
    source_dataset = openpi_data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    transformed_dataset = openpi_data_loader.transform_dataset(
        source_dataset,
        data_config,
        skip_norm_stats=False,
    )

    sample_index = int(args.sample_index)
    source_sample = source_dataset[sample_index]
    transformed_sample = transformed_dataset[sample_index]

    _print_section("Source Sample")
    _summarize_sample(source_sample, prefix="source")

    _print_section("Transformed Training Sample")
    _summarize_sample(transformed_sample, prefix="train_sample")

    model = get_model(cfg.actor.model, torch_dtype=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _print_section("Model Action Config")
    print(f"model.config.action_chunk={getattr(model.config, 'action_chunk', None)}")
    print(f"model.config.action_horizon={getattr(model.config, 'action_horizon', None)}")
    print(f"model.config.action_dim={getattr(model.config, 'action_dim', None)}")
    print(f"model.config.action_env_dim={getattr(model.config, 'action_env_dim', None)}")
    print(f"model.config.num_steps={getattr(model.config, 'num_steps', None)}")

    train_observation, train_actions = _prepare_training_batch(
        transformed_sample, device
    )

    _print_section("Training Batch")
    _summarize_observation(train_observation)
    _stats("training.actions", train_actions)

    _print_section("Initial SFT Loss")
    model.eval()
    with torch.no_grad():
        initial_loss, initial_output = _mean_sft_loss(
            model, train_observation, train_actions
        )
    print(f"sft_loss_mean={initial_loss.detach().cpu().item():.8f}")
    if isinstance(initial_output, dict):
        for key, value in initial_output.items():
            if torch.is_tensor(value):
                _stats(f"sft_output[{key}]", value)
            else:
                print(f"sft_output[{key}]={value}")
    else:
        _stats("sft_output", initial_output)

    ground_truth_actions = _get_raw_gt_actions(source_sample)
    _print_section("Initial Model Output Action vs Ground Truth Action")
    model_output_actions, result = _predict_actions(
        model, source_sample, seed=args.seed
    )
    _stats("model_output.actions", model_output_actions)
    _stats("ground_truth.actions", ground_truth_actions)
    _compare_model_output_to_ground_truth(model_output_actions, ground_truth_actions)

    _fit_single_sample(
        model,
        train_observation,
        train_actions,
        steps=args.fit_steps,
        lr=args.fit_lr,
        log_interval=args.fit_log_interval,
    )

    if args.fit_steps > 0:
        _print_section("Final SFT Loss")
        model.eval()
        with torch.no_grad():
            final_loss, _ = _mean_sft_loss(model, train_observation, train_actions)
        print(f"sft_loss_mean={final_loss.detach().cpu().item():.8f}")

        _print_section("Final Model Output Action vs Ground Truth Action")
        model_output_actions, result = _predict_actions(
            model, source_sample, seed=args.seed
        )
        _stats("model_output.actions", model_output_actions)
        _stats("ground_truth.actions", ground_truth_actions)
        _compare_model_output_to_ground_truth(model_output_actions, ground_truth_actions)

    forward_inputs = result.get("forward_inputs", {})
    if forward_inputs:
        _print_section("Forward Inputs")
        for key, value in forward_inputs.items():
            if torch.is_tensor(value):
                _stats(f"forward_inputs[{key}]", value)


if __name__ == "__main__":
    main()
