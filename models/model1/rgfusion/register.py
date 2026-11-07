"""Plug Model 1 into OpenPCDet without editing OpenPCDet's source.

Import this module before OpenPCDet builds a dataset or model:
  - 'Model1Fusion' is added to pcdet.models.detectors.__all__ (looked up by
    build_detector at call time).
  - 'model1_modality_dropout' becomes a DataAugmentor method (DataAugmentor
    resolves AUG_CONFIG_LIST names with getattr). Dataloader workers are forked
    after this import, so they inherit it.
  - If M1_EVAL_CORRUPTION=<module>:<function> is set, that function is applied
    to every TEST sample before voxelisation: the hook the corruption engine
    plugs into. Signature: fn(data_dict, rng) -> data_dict. Never applied in
    training (context.md section 6).
"""

import importlib
import os
import zlib

import numpy as np

from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.models import detectors

from .detector import Model1Fusion
from .modality_dropout import model1_modality_dropout

detectors.__all__["Model1Fusion"] = Model1Fusion
DataAugmentor.model1_modality_dropout = model1_modality_dropout


def _install_eval_corruption(spec: str) -> None:
    module_name, fn_name = spec.split(":")
    fn = getattr(importlib.import_module(module_name), fn_name)
    original = DatasetTemplate.prepare_data

    def prepare_data(self, data_dict):
        if not self.training:
            # Seeded by frame id: the same frame gets the same corruption on every run.
            seed = zlib.crc32(str(data_dict.get("frame_id", "")).encode())
            data_dict = fn(data_dict, np.random.default_rng(seed))
        return original(self, data_dict)

    DatasetTemplate.prepare_data = prepare_data


if os.environ.get("M1_EVAL_CORRUPTION"):
    _install_eval_corruption(os.environ["M1_EVAL_CORRUPTION"])
