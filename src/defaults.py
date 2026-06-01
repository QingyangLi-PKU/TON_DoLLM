"""Default hyperparameters aligned with DoLLM_bert."""

from __future__ import annotations

# train.py / zeroshot_test_multiseed.py
SEED = 42
SEEDS = [42, 123, 456]

BATCH_SIZE = 64
NUM_WORKERS = 4
BINNING_NUM = 64
NUM_TRAINING_SAMPLES = 15000  # num_traning_samples in original code

NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATE_EPOCH_INTERVAL = 1

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

PROJECTION_OUTPUT_DIM = 256
OUTPUT_CLASSES = 2
DROPOUT = 0.0

# zeroshot_test_multiseed.py uses BERT; train.py active path used roberta-base
DEFAULT_MODEL_PATH = "/mnt/ssd1/model_zoo/bert-base-uncased"
TRAIN_SCRIPT_MODEL_PATH = "/mnt/ssd1/model_zoo/roberta-base"

# Bundled datasets (cic_single_vec ratio_1:3, same as zeroshot root subsets)
DEFAULT_SYN_SRC = "/mnt/ssd1/DoLLM_TON_dataset/eva_dataset/cic_single_vec/ratio_1:3/Syn"
DEFAULT_UDP_SRC = "/mnt/ssd1/DoLLM_TON_dataset/eva_dataset/cic_single_vec/ratio_1:3/UDPLag"
DEFAULT_TOY_ROOT = "toy_datasets"
