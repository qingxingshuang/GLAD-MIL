# GLAD-MIL

Public code release for GLAD-MIL, a dual-granularity multiple-instance learning method for whole-slide image (WSI) classification from pre-extracted patch features.

This repository is the code-only release accompanying the GLAD-MIL study. It makes the model architecture, training protocol, evaluation procedure, and input-file contract available for independent research use.

**Release scope:** source code and runnable configuration templates are included; trained weights and study data are intentionally excluded.

## Contents

The model retains a full-bag LGSE-TransMIL decision route and adds randomly partitioned pseudo-bags. The pseudo-bag class-token descriptors are re-aggregated with gated attention. Training uses `CE(full) + alpha * CE(pseudo-bags) + beta * CE(tier2)`. The checkpoint is selected by validation AUC, with validation loss as a tie-break. After checkpoint selection, `gamma` is selected on the validation set and frozen for test evaluation:

`P(y=1) = (1 - gamma) * P_full(y=1) + gamma * P_tier2(y=1)`.

The public release contains no trained weights, WSI data, patch-feature tensors, patient information, study-specific manifests or split files, or third-party baseline code. The example manifest is a schema example only and does not contain study records.

## What is released

- The GLAD-MIL model, including the LGSE--TransMIL backbone, pseudo-bag construction, second-tier gated aggregation, joint supervision, and validation-selected prediction fusion.
- The training and evaluation entry point, metric implementation, configuration template, and manifest format.
- A minimal interface for users to supply their own pre-extracted feature files.

## What is required from the user

The code expects one pre-extracted feature tensor per slide and a user-provided manifest. It does not download or redistribute WSIs, patch features, patient metadata, pretrained encoder weights, or the frozen study split. Reproducing the numerical results reported in the paper therefore requires access to the same feature release and split protocol described in the paper, in addition to the code in this repository.

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Data manifest

Provide a CSV with `slide_id,split,label,feature_path`; see `examples/manifest_example.csv`. `split` must be `train`, `val`, or `test`. Each feature path must point to a `[number_of_patches, feature_dimension]` tensor saved as `.pt`, a tensor nested under `features`, `feature`, or `tensor` in a `.pt` dictionary, or a `.npy` array.

Paths are resolved relative to the manifest file. Feature files are intentionally ignored by Git.

## Run

Copy `configs/example.yaml`, change `manifest`, `output_dir`, and `input_dim`, then run. Paths in the configuration are resolved relative to the configuration file:

```bash
python -m glad_mil.train --config configs/example.yaml
```

The command writes `best.pt` and `metrics.json` under `output_dir`. Those run-specific artifacts are excluded from source control.

## Reproducibility

Set `seed` in the YAML file. This public implementation enables deterministic PyTorch settings where supported. Supply the same frozen feature files and data split used for the desired experiment. Dataset access, feature extraction, pretrained encoders, and all upstream licenses remain the responsibility of the user; this repository neither redistributes nor grants rights to them.

Training uses a fresh seeded random pseudo-bag partition for each slide update. Validation and testing use one deterministic partition: feature-row order is split into contiguous groups. Gamma ties are resolved by choosing the value closest to 0.5, then the smaller value.

## Citation and attribution

If you use this code, please cite the GLAD-MIL paper associated with the release. The permanent repository URL can be included in the paper and release metadata once the hosting address is fixed. The GLAD-MIL source code is released under the MIT License; PyTorch, NumPy, PyYAML, the datasets, feature extractors, and other upstream resources remain subject to their respective licenses and terms.

## License

MIT. See `LICENSE`.
