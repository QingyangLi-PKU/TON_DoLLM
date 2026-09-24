## Repository layout

```text
DoLLM_minimal_open/
├── src/                         # DoLLM implementation
│   ├── data.py                  # flow preprocessing and FS construction
│   ├── defaults.py              # experiment hyperparameters
│   ├── model.py                 # DoLLM model
│   └── train_eval.py            # training and evaluation routines
├── scripts/
│   └── run_experiments.py       # one-command experiment entry point
├── toy_datasets/
│   ├── Syn/
│   └── UDP/                    
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <repository-url>
cd DoLLM_minimal_open
conda create -n dollm -c pytorch -c conda-forge --file requirements.txt -y
conda activate dollm
```

By default, experiments use the local LM checkpoint
`/mnt/ssd1/model_zoo/roberta-base`. Supply `--model-name-or-path` to use a
different local checkpoint or a Hugging Face model identifier.


## One-command reproduction

Run the following command without any arguments:

```bash
python scripts/run_experiments.py
```

It runs all bundled results with the default seed `42`:

- Syn in-domain evaluation.
- UDP in-domain evaluation.
- `Syn -> UDP` — zeroshot.
- `UDP -> Syn` — zeroshot.

Results are written to `outputs/experiments/`:

To run the complete suite with multiple seeds, for example:

```bash
python scripts/run_experiments.py --seeds 42 123 456
```

## Default experiment settings

| Parameter | Value |
|---|---:|
| Backbone | frozen `roberta-base` |
| Flow sequence length | 64 |
| Training FS samples | 15,000 |
| Batch size | 64 |
| Epochs | 20 |
| Learning rate | 1e-4 |
| Loss | focal loss (`alpha=0.25`, `gamma=2`) |
| Default seed | 42 |
