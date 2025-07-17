#-*- coding: utf-8 -*-
import os
import sys
from typing import Any, Optional, Dict

# Ensure current working directory is in path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# Import the registry and factory function
from register.register import Registry, build_from_cfg

# Define registries for Pipelines and Datasets
PIPELINES = Registry('Pipeline', build_func=build_from_cfg)
DATASETS = Registry('Dataset', build_func=build_from_cfg)

# Function to build a pipeline object from config
def build_pipeline(cfg,
                   pipeline: Registry = PIPELINES,
                   build_func=build_from_cfg,
                   default_args: Optional[Dict] = None) -> Any:
    return build_func(cfg, pipeline, default_args)

# Function to build a dataset object from config
def build_dataset(cfg,
                  dataset: Registry = DATASETS,
                  build_func=build_from_cfg,
                  default_args: Optional[Dict] = None) -> Any:
    return build_func(cfg, dataset, default_args)
