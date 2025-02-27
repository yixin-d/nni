# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest

import functools
from typing import Callable, Union, List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor

import nni
from nni.contrib.compression.quantization import BNNQuantizer
from nni.contrib.compression.utils import TorchEvaluator

from ..assets.simple_mnist import (
    SimpleTorchModel, 
    SimpleLightningModel,
    create_lighting_evaluator,
    create_pytorch_evaluator,
)
from ..assets.device import device


def test_bnn_forward_with_torch_model():
    torch.manual_seed(0)
    model = SimpleTorchModel().to(device)
    configure_list = [{
        'target_names':['_input_', 'weight', '_output_'],
        'op_names': ['fc1', 'fc2'],
        'quant_dtype': None,
        'quant_scheme': 'affine',
        'granularity': 'default',
    },
    {
        'target_names':['_input_', 'weight', '_output_'],
        'op_names': ['conv1', 'conv2', 'conv3'],
        'quant_dtype': None,
        'quant_scheme': 'affine',
        'granularity': 'default',
        'fuse_names': [("conv1", "bn1"), ('conv2', 'bn2'), ('conv3', 'bn3')]
    }]
    evaluator = create_pytorch_evaluator(model)
    quantizer = BNNQuantizer(model, configure_list, evaluator)
    quantizer.compress(None, 1)


def test_bnn_forward_with_lighting_model():
    torch.manual_seed(0)
    configure_list = [{
        'target_names':['_input_', 'weight', '_output_'],
        'op_names': ['model.fc1', 'model.fc2'],
        'quant_dtype': None,
        'quant_scheme': 'affine',
        'granularity': 'default',
    },
    {
        'target_names':['_input_', 'weight', '_output_'],
        'op_names': ['model.conv1', 'model.conv2', 'model.conv3'],
        'quant_dtype': None,
        'quant_scheme': 'affine',
        'granularity': 'default',
        'fuse_names': [("model.conv1", "model.bn1"), ('model.conv2', 'model.bn2'), ('model.conv3', 'model.bn3')]
    }]
    evaluator = create_lighting_evaluator()
    model = SimpleLightningModel().to(device)
    quantizer = BNNQuantizer(model, configure_list, evaluator)
    quantizer.compress(None, 1)
