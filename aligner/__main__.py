#!/usr/bin/env python
# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import logging
import os
from time import strftime
from pathlib import Path
from typing import Mapping, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import NeptuneLogger

from hydra.utils import to_absolute_path

from aligner.cli import create_model_data_module_trainer, init_cli
from aligner.utils.logger_utils import get_logger_by_type

try:
    import wandb
except ImportError:
    wandb = None

LOGGER = logging.getLogger(__name__)

os.environ.setdefault("SWEEP_DIR", f"multirun/{strftime('%Y-%m-%d')}/{strftime('%H-%M-%S')}")

@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig) -> Optional[float]:
    init_cli(cfg)
    
    if cfg.wandb.enabled:
        assert wandb is not None, 'Please install wandb.'
        wandb.init(
            entity='#YOUR_WANDB_ENTITY',
            project=cfg.wandb.project_name,
            name=f'{Path(cfg.output_dir).parts[-1]}__{cfg.data.dataset_name}',
        )
    
    if not os.path.isdir(to_absolute_path(cfg.output_dir)):
        os.makedirs(to_absolute_path(cfg.output_dir))

    if cfg.get("trainer", {}).get("strategy") == "dp":
        LOGGER.warning("DP strategy not supported by the current metric logging scheme."
                       " See https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#logging-torchmetrics")

    model, data_module, trainer = create_model_data_module_trainer(cfg)

    if cfg.command in {"evaluate", "validate"}:
        with torch.inference_mode():
            trainer.validate(model, datamodule=data_module)
    else:
        raise ValueError(f"Unrecognized command: {cfg.command}")

    if (neptune_logger := get_logger_by_type(trainer, NeptuneLogger)) and trainer.is_global_zero:
        neptune_logger.run.stop()
        
if __name__ == "__main__":
    main()
