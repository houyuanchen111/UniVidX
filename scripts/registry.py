import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.pipelines.univid_alpha import UniVidAlpha
from src.pipelines.univid_intrinsic import UniVidIntrinsic
from src.trainers.unividx_trainer import ModelCheckpointCallback, TensorboardLoggingCallback, Trainer


DATASET_REGISTRY = {
    # register your dataset here
}


MODEL_REGISTRY = {
    'UniVidAlpha': UniVidAlpha,
    'UniVidIntrinsic': UniVidIntrinsic
}

OPTIMIZER_REGISTRY = {
    'AdamW': AdamW,
}

SCHEDULER_REGISTRY = {
    'CosineAnnealingLR': CosineAnnealingLR,
}

CALLBACK_REGISTRY = {
    'ModelCheckpointCallback': ModelCheckpointCallback,
    'TensorboardLoggingCallback': TensorboardLoggingCallback,
}   


TRAINER_REGISTRY = {
    'Trainer': Trainer,
}

