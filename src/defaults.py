"""Default hyperparameters """

from __future__ import annotations

SEED = 42

SEEDS = [SEED]

BATCH_SIZE = 64
NUM_WORKERS = 4
BINNING_NUM = 64
NUM_TRAINING_SAMPLES = 15000  

NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATE_EPOCH_INTERVAL = 1

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

PROJECTION_OUTPUT_DIM = 256
OUTPUT_CLASSES = 2
DROPOUT = 0.0

# Default checkpoint used by the original local DoLLM experiments.
DEFAULT_MODEL_PATH = "/mnt/ssd1/model_zoo/roberta-base"
