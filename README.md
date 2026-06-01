# DoLLM Minimal Open

- **SYN** and **UDP** datasets (bundled toy splits under `toy_datasets/`)
- **In-domain** training and evaluation
- **Zero-shot** cross-dataset evaluation (`SYN → UDP`, `UDP → SYN`)
- Consistent naming: the evaluation split is called 

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use a local LMs checkpoint if available (recommended):

```bash
export DOLLM_MODEL=/mnt/ssd1/model_zoo/roberta-base
```

## Bundled SYN / UDP datasets

Bundled CSVs live in `toy_datasets/Syn` and `toy_datasets/UDP`. The copy script uses the full source CSV splits by default and does not change the original DoLLM hyperparameters. Each folder contains:

| File | Role |
|------|------|
| `mixed_flows_train.csv` | Training |
| `mixed_flows_valid.csv` | Validation (model selection) |
| `mixed_flows_current_detection_batch.csv` | Held-out current detection batch |


## In-domain train + evaluate

Train on SYN and evaluate on its **current_detection_batch**:

```bash
PYTHONPATH=src python scripts/train.py \
  --dataset-dir toy_datasets/Syn \
  --model-name-or-path "$DOLLM_MODEL" \
  --output-dir outputs/syn_in_domain \
  --seed 42
```

Outputs:

- `outputs/syn_in_domain/best_model.pt`
- `outputs/syn_in_domain/in_domain_metrics.npy`

## Zero-shot (SYN ↔ UDP)

```bash
PYTHONPATH=src python scripts/evaluate_zero_shot.py \
  --syn-dir toy_datasets/Syn \
  --udp-dir toy_datasets/UDP \
  --model-name-or-path "$DOLLM_MODEL" \
  --output-dir outputs/syn_udp_zero_shot \
  --seeds 42
```

Results:

- `zero_shot_results.csv` — per seed / source / target
- `zero_shot_summary.csv` — mean/std over zero-shot cells

| `setting` | Meaning |
|-----------|---------|
| `in_domain` | `source_dataset == target_dataset` |
| `zero_shot` | `source_dataset != target_dataset` |

## Default hyperparameters

| Parameter | Default |
|-----------|---------|
| `batch_size` | `64` |
| `num_workers` | `4` |
| `binning_num` / `num_flows` | `64` |
| `num_training_samples` | `15000` |
| `num_epochs` | `20` |
| `learning_rate` | `1e-4` |
| `seeds` | `42` |
| zero-shot model | `/mnt/ssd1/model_zoo/roberta-base` |
| single-train model | `/mnt/ssd1/model_zoo/roberta-base` |

