"""Hydra-free entry point for train.py. Edit the knobs in `load_cfg` and run:

    python train_no_hydra.py
"""
import os
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import lightning as pl
import stable_pretraining as spt  # also registers the ${eval:...} resolver used in data/*.yaml
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from train import (
    FixedCoverageMaxTransitionsBatchSampler,
    FixedStartVariableTransitionsDataset,
    _episode_column_name,
    _resolve_alignment_mode,
    lejepa_forward,
)
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack

CONFIG_DIR = Path(__file__).parent / "config" / "train"


def load_cfg(data: str = "pusht"):
    cfg = OmegaConf.load(CONFIG_DIR / "Fast-lewm.yaml")
    cfg.pop("defaults")
    cfg.data = OmegaConf.load(CONFIG_DIR / "data" / f"{data}.yaml")
    cfg.subdir = datetime.now().strftime("%Y%m%d-%H%M%S")  # replaces ${hydra:job.id}

    # --- knobs ---------------------------------------------------------------
    # RAM in flight ~= num_workers * prefetch_factor * batch_size
    #                  * (max_transitions * action_frame_skip + 1) frames * 602 KB
    cfg.loader.batch_size = 16
    cfg.num_workers = 6
    cfg.loader.prefetch_factor = 3
    cfg.wm.action_frame_skip = 5
    cfg.wm.dense_supervision.max_transitions = 5
    cfg.wandb.enabled = True
    return cfg


def build_loaders(cfg):
    """Returns (train, val) loaders. Also fills cfg.wm.*_dim, so call before build_model."""
    #########################
    ##       dataset       ##
    #########################
    pl.seed_everything(int(cfg.seed), workers=True)

    dense_cfg = cfg.wm.get("dense_supervision", None)
    action_frame_skip = int(cfg.wm.get("action_frame_skip", 1))
    if action_frame_skip < 1:
        raise ValueError(
            f"wm.action_frame_skip must be >= 1, got {action_frame_skip}."
        )

    base_dataset_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    if "num_steps" in base_dataset_kwargs:
        configured_num_steps = int(base_dataset_kwargs["num_steps"])
        if configured_num_steps < 2:
            raise ValueError(
                f"data.dataset.num_steps must be >= 2, got {configured_num_steps}."
            )
        base_dataset_kwargs["num_steps"] = (
            (configured_num_steps - 1) * action_frame_skip + 1
        )
    if "keys_to_cache" in base_dataset_kwargs:
        # In max_adaptive mode, base_dataset is only used for stats/normalization.
        base_dataset_kwargs["keys_to_cache"] = []

    base_dataset_kwargs['cache_dir'] = os.path.expanduser('~/stablewm_data/')
    base_dataset = swm.data.HDF5Dataset(**base_dataset_kwargs, transform=None)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(base_dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", base_dataset.get_dim(col))

        if hasattr(cfg.wm, "action_dim"):
            base_action_dim = int(cfg.wm.action_dim)
            cfg.wm.action_base_dim = base_action_dim
            cfg.wm.action_dim = int(base_action_dim * action_frame_skip)
        cfg.wm.action_frame_skip = int(action_frame_skip)

    transform = spt.data.transforms.Compose(*transforms)
    base_dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(int(cfg.seed))

    min_transitions = int(dense_cfg.get("min_transitions", 1))
    max_transitions = int(dense_cfg.get("max_transitions", min_transitions))

    dataset_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    reference_transitions = int(min_transitions)
    aux_keys_to_cache_cfg = dense_cfg.get("aux_keys_to_cache", [])
    aux_keys_to_cache = (
        None if aux_keys_to_cache_cfg is None else list(aux_keys_to_cache_cfg)
    )
    requested_transitions = list(range(min_transitions, max_transitions + 1))
    datasets_by_transitions = {}
    for transitions in requested_transitions:
        cur_kwargs = dict(dataset_kwargs)
        cur_kwargs["num_steps"] = int(transitions * action_frame_skip + 1)
        if (
            transitions != reference_transitions
            and aux_keys_to_cache is not None
            and "keys_to_cache" in cur_kwargs
        ):
            cur_kwargs["keys_to_cache"] = aux_keys_to_cache
        datasets_by_transitions[transitions] = swm.data.HDF5Dataset(
            **cur_kwargs,
            transform=transform,
        )

    reference_dataset = datasets_by_transitions[reference_transitions]
    train_subset, _ = spt.data.random_split(
        reference_dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    val_source_dataset = datasets_by_transitions[int(max_transitions)]
    val_gen = torch.Generator().manual_seed(int(cfg.seed) + 1)
    _, val_set = spt.data.random_split(
        val_source_dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=val_gen,
    )

    base_mode = _resolve_alignment_mode(reference_dataset)
    episode_col = _episode_column_name(reference_dataset)
    if base_mode == "episode_step":
        episode_idx = reference_dataset.get_col_data(episode_col)
        step_idx = reference_dataset.get_col_data("step_idx")
        base_keys = [
            (int(episode_idx[i]), int(step_idx[i]))
            for i in train_subset.indices
        ]
    else:
        base_keys = [(0, int(i)) for i in train_subset.indices]

    train_set = FixedStartVariableTransitionsDataset(
        datasets_by_transitions=datasets_by_transitions,
        base_keys=base_keys,
        fallback_transitions=reference_transitions,
    )

    resolved_transitions_by_local = [
        train_set.resolve_local_transitions(i, max_transitions)
        for i in range(len(train_set))
    ]
    resolution_hist = defaultdict(int)
    for transitions in resolved_transitions_by_local:
        resolution_hist[int(transitions)] += 1
    hist_summary = ", ".join(
        f"{t}:{resolution_hist[t]}" for t in sorted(resolution_hist.keys())
    )
    print(
        "Dense transition max-adaptive buckets (transitions:count): "
        f"{hist_summary}"
    )

    batch_size = int(cfg.loader.batch_size)
    drop_last = bool(cfg.loader.get("drop_last", True))
    train_batch_sampler = FixedCoverageMaxTransitionsBatchSampler(
        num_samples=len(train_set),
        batch_size=batch_size,
        resolved_transitions_by_local=resolved_transitions_by_local,
        drop_last=drop_last,
        seed=int(cfg.seed),
    )

    train_loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    train_loader_kwargs.pop("batch_size", None)
    train_loader_kwargs.pop("shuffle", None)
    train_loader_kwargs.pop("drop_last", None)

    train = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=train_batch_sampler,
        generator=rnd_gen,
        **train_loader_kwargs,
    )

    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )
    return train, val


def build_model(cfg):
    ##############################
    ##          model           ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)

    predictor = ARPredictor(
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_embedder_kwargs = {
        "use_positional_encoding": bool(
            cfg.wm.get("action_use_positional_encoding", True)
        )
    }
    action_prefix_cfg = cfg.wm.get("action_prefix", None)
    if action_prefix_cfg is not None:
        action_embedder_kwargs.update(
            OmegaConf.to_container(action_prefix_cfg, resolve=True)
        )

    action_encoder = Embedder(
        input_dim=cfg.wm.action_dim,
        emb_dim=embed_dim,
        **action_embedder_kwargs,
    )
    action_encoder_impl = getattr(action_encoder, "impl", action_encoder)
    setattr(action_encoder_impl, "action_base_dim", int(cfg.wm.action_base_dim))
    setattr(action_encoder_impl, "action_frame_skip", int(cfg.wm.action_frame_skip))

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )
    return world_model


def run(cfg, run_dir: Path):
    train, val = build_loaders(cfg)
    world_model = build_model(cfg)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        default_root_dir=str(run_dir),
        enable_checkpointing=True,
        use_distributed_sampler=False,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()


if __name__ == "__main__":
    cfg = load_cfg("pusht")
    run(cfg, Path("outputs") / cfg.subdir)
