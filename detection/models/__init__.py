# Copyright (c) Facebook, Inc. and its affiliates.
from .model_3detr import build_3detr
from .model_3detr_lcm import build_3detr_lcm
from .model_3detr_t import build_3detr_t

MODEL_FUNCS = {
    "3detr": build_3detr,
    "3detr_lcm": build_3detr_lcm,
    "3detr_t": build_3detr_t,
}

def build_model(args, dataset_config):
    model, processor = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model, processor