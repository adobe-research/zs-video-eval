#!/usr/bin/env python
# Copyright 2025 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import copy
import logging
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Type

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from . import TextVideoGroundingLightningModule
from aligner.data.data_module_group import DataModuleStructuredGroup, EvalDataModuleGroup, \
                                            MixedBatchDataModule, TrainAndEvalDataModules
from aligner.data.video_data_module import ENCODER_OR_ENCODER_MAP
from aligner.data.grounding import ActivityNetCaptionsDataModule, CharadesSTADataModule

LOGGER = logging.getLogger(__name__)

# This is because PL can't automatically infer the batch size, that's needed for logging. But we set it manually
# within the modules.
warnings.filterwarnings("ignore", message=r"^Trying to infer the `batch_size` from an ambiguous collection\. .+")


# From https://stackoverflow.com/a/2020083/1165181
def fullname(klass: Type[Any]) -> str:
    return f"{klass.__module__}.{klass.__qualname__}"


def set_logging_level(level: int) -> None:
    logging.basicConfig(level=level)
    # `basicConfig` will only work for new loggers, so we also need to set up the existing ones:
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):  # Otherwise, it could be a `logging.PlaceHolder`.
            logger.setLevel(level)
    logging.getLogger().setLevel(level)  # The root logger is not present in the previous iterable.


def init_cli(cfg: DictConfig) -> None:
    if cfg.get("silent"):
        set_logging_level(logging.WARNING)
    else:
        set_logging_level(logging.INFO)

    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)


def instantiate_data_module(cfg: DictConfig, encoder: ENCODER_OR_ENCODER_MAP) -> pl.LightningDataModule:
    kwargs = {}

    if cfg._target_ in {fullname(klass) for klass in [DataModuleStructuredGroup, EvalDataModuleGroup,
                                                      MixedBatchDataModule]}:
        if isinstance(cfg.data_modules, Mapping):
            kwargs["data_modules"] = {k: instantiate_data_module(v, encoder=encoder)  # noqa
                                      for k, v in cfg.data_modules.items()}
        else:
            kwargs["data_modules"] = {instantiate_data_module(cfg_dm, encoder=encoder)
                                      for cfg_dm in cfg.data_modules}

        # Convert because otherwise the passed `data_modules` is a `DictConfig` instead of a `dict` and
        # `train_dataloader` can't respect the same collection type as `DictConfig` can't have normal classes.
        kwargs["_convert_"] = "all"
    elif cfg._target_ == fullname(TrainAndEvalDataModules):
        kwargs["train_data_module"] = instantiate_data_module(cfg.train_data_module, encoder=encoder)

        kwargs["eval_data_module"] = instantiate_data_module(cfg.eval_data_module, encoder=encoder)
    else:
        kwargs["encoder"] = encoder

        # Necessary as well when the encoder is a dict.
        kwargs["_convert_"] = "all"

    return hydra.utils.instantiate(cfg, **kwargs)


def create_model_data_module_trainer(
        cfg: DictConfig, model_kwargs: Optional[Mapping[str, Any]] = None) -> Tuple[TextVideoGroundingLightningModule,
                                                                                    pl.LightningDataModule, pl.Trainer,
                                                                                    str]:
    model_kwargs = model_kwargs or {}

    LOGGER.info(f"Instantiating encoder <{getattr(cfg.encoder, '_target_', type(cfg.encoder).__name__)}>…")
    if cfg.encoder.model.use_residual_feat is True:
        cfg.encoder.use_residual_feat = True
    encoder: ENCODER_OR_ENCODER_MAP = hydra.utils.instantiate(cfg.encoder)
    LOGGER.info("Encoder instantiated.")

    LOGGER.info(f"Instantiating data module <{cfg.data._target_}>…")
    data_module = instantiate_data_module(cfg.data, encoder=encoder)
    LOGGER.info("Data module instantiated.")

    LOGGER.info(f"Instantiating model <{cfg.model._target_}>…")
    model_kwargs.setdefault("encoder", encoder)

    if isinstance(data_module, (ActivityNetCaptionsDataModule, CharadesSTADataModule)):
        
        cfg.model._target_ = fullname(TextVideoGroundingLightningModule)
        model_kwargs.setdefault("datamodule", data_module)
        model_kwargs.setdefault("watershed_threshold", cfg.post_processing.watershed_threshold)
        model_kwargs.setdefault("watershed_scoring", cfg.post_processing.watershed_scoring)
        model_kwargs.setdefault("smoothing_function", cfg.post_processing.smoothing_function)
        model_kwargs.setdefault("smoothing_window_size", cfg.post_processing.smoothing_window_size)
        model_kwargs.setdefault("output_dir", cfg.output_dir)
        model_kwargs.setdefault("dataset_name", cfg.data.dataset_name)
        model_kwargs.setdefault("patch_drop", cfg.encoder.model.force_patch_dropout)
        model_kwargs.setdefault("patch_merge", cfg.encoder.model.patch_merging_r)
        try:
            model_kwargs.setdefault("evaluation_epoch", str(Path(cfg.encoder.model.pretrained).stem).split('_')[-1])
        except:
            '''If we are evaluating a CLIP model'''
            model_kwargs.setdefault("evaluation_epoch", 0)

        model_name = f"MODEL_{cfg.encoder['model']['name']}" 
        if 'distill_model' in cfg.encoder['model'].keys() and cfg.encoder['model']['distill_model'] is not None:
            model_name += f"_DISTILL_{cfg.encoder['model']['distill_model']}"
        model_kwargs.setdefault("model_name", model_name)
        model_kwargs.setdefault("enable_wandb", cfg.wandb.enabled)

    else:
        raise ValueError(f"Unrecognized data module: {data_module}")

    model: TextVideoGroundingLightningModule = hydra.utils.instantiate(cfg.model, **model_kwargs)
    LOGGER.info("Model instantiated.")

    LOGGER.info(f"Instantiating trainer <{cfg.trainer._target_}>…")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    LOGGER.info("Trainer instantiated.")

    model._log_hyperparams = trainer.logger
    model._set_hparams(cfg)  # noqa
    model._hparams_initial = copy.deepcopy(model._hparams)
    data_module.output_dir = cfg.output_dir

    return model, data_module, trainer
