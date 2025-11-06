# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DualCnst (formerly MCM - Maximum Concept Matching) is a research codebase for out-of-distribution (OOD) detection using vision-language models, particularly CLIP. The project implements the paper "Delving into Out-Of-Distribution Detection with Vision-Language Representations" (NeurIPS 2022).

## Environment Setup

```bash
# Create conda environment from specification
conda env create -f sd.yml
conda activate sd
```

The environment uses Python 3.10.18, PyTorch 2.7.1+cu118, and CLIP from OpenAI. Key dependencies include transformers, torchvision, diffusers, and modelscope.

## Common Commands

### Creating ImageNet Subsets

```bash
# ImageNet-10 (mimics CIFAR-10 class distribution)
python create_imagenet_subset.py --in_dataset ImageNet10 --src-dir datasets/ImageNet --dst-dir datasets

# ImageNet-20 (semantically similar to ImageNet-10, for hard OOD)
python create_imagenet_subset.py --in_dataset ImageNet20 --src-dir datasets/ImageNet --dst-dir datasets

# ImageNet-100
python create_imagenet_subset.py --in_dataset ImageNet100 --src-dir datasets/ImageNet --dst-dir datasets
```

### Running OOD Detection Evaluation

```bash
# Basic evaluation command
python eval_ood_detection.py --score MCM --in_dataset ImageNet --gpu 0 --name <experiment_name>

# Using the shell script wrapper
sh scripts/eval_dualcnst.sh <run_name> <in_dataset> <score_type>

# Example: evaluate DualCnst score on ImageNet
sh scripts/eval_dualcnst.sh eval_ood ImageNet DualCnst

# Quick smoke test with small batch
python eval_ood_detection.py --batch_size 2 --in_dataset ImageNet10 --gpu 0 --name smoke-test
```

### Important Script Arguments

- `--score`: Detection score type (`MCM`, `DualCnst`, `NegLabel`, `energy`, `max-logit`, `entropy`, `var`, `maha`)
- `--in_dataset`: In-distribution dataset (`ImageNet`, `ImageNet10`, `ImageNet20`, `ImageNet100`, `bird200`, `car196`, `food101`, `pet37`)
- `--CLIP_ckpt`: CLIP model variant (`ViT-B/32`, `ViT-B/16`, `ViT-L/14`)
- `--T`: Temperature parameter (default: 0.01)
- `--alpha`: Weight parameter for image-image similarity (default: 0.2)
- `--batch_size`: Mini-batch size (default: 1024)
- `--selected_layers`: Comma-separated layer indices for intermediate features (e.g., `7,10`)
- `--layer_weight`: Weight for intermediate layer features (default: 0.15)
- `--cache_dir`: Root directory for cache files (default: `similarity_caches`)
- `--cache_path`: Override for explicit cache path (auto-generated if not provided)
- `--intermediate_cache_path`: Override for intermediate cache path (auto-generated if not provided)

### Automatic Path Generation (NEW!)

**MAJOR UPDATE**: The codebase now automatically generates ALL paths based on dataset configuration:
- ✅ **Positive/Negative label paths** (`pos_path`, `neg_path`)
- ✅ **Cache paths** (`cache_path`, `intermediate_cache_path`)

This completely eliminates manual path modification when switching datasets!

#### Label Path Auto-generation

**Format**: `{neglabel_dir}/{subdir}/{positive|negative}/{filename}`

**Examples**:
```bash
# ImageNet
pos_path: OODNegMining/ImageNet/positive/ImageNet.txt
neg_path: OODNegMining/ImageNet/negative/ImageNet_neg.txt

# ImageNet10
pos_path: OODNegMining/ImageNet/positive/ImageNet10.txt
neg_path: OODNegMining/ImageNet/negative/ImageNet10_neg.txt

# bird200
pos_path: OODNegMining/CUB/positive/positive_samples_bird200.txt
neg_path: OODNegMining/CUB/negative/low_similarity_neg_samples_bird200.txt
```

#### Cache Path Auto-generation

**Format**: `{cache_dir}/{in_dataset}/{model}/layers_{layers}/{in_dataset}_{feature_type}`

**Examples**:
```bash
# ImageNet with ViT-B/16, layers 7,10
cache_path: similarity_caches/ImageNet/ViT-B-16/layers_7_10/ImageNet_final_features.npy
intermediate_cache_path: similarity_caches/ImageNet/ViT-B-16/layers_7_10/ImageNet_intermediate_features.pt

# ImageNet100 with ViT-L/14, layers 1,13
cache_path: similarity_caches/ImageNet100/ViT-L-14/layers_1_13/ImageNet100_final_features.npy
intermediate_cache_path: similarity_caches/ImageNet100/ViT-L-14/layers_1_13/ImageNet100_intermediate_features.pt
```

#### Supported Datasets

Auto-path generation works for all standard datasets:
- **ImageNet variants**: `ImageNet`, `ImageNet10`, `ImageNet20`, `ImageNet100`, `ImageNet_R`, `ImageNet_A`, etc.
- **Fine-grained datasets**: `bird200` (CUB), `food101` (Food), `car196` (Stanford-Cars), `pet37` (Oxford-Pet)

See `docs/cache_path_guide.md` for detailed documentation.

## Architecture Overview

### Core Entry Point

`eval_ood_detection.py` is the main orchestration script that:
1. Loads CLIP models and datasets
2. Computes OOD scores using various methods
3. Evaluates detection performance (AUROC, AUPR, FPR95)
4. Saves results to `results/<dataset>/<score>/<model>_<ckpt>_T_<T>_ID_<name>/`

Note: Line 2 hardcodes GPU device `CUDA_VISIBLE_DEVICES = "7"` - modify this for your setup.

### Module Organization

- **`dataloaders/`**: Custom dataset classes for CUB-200, Stanford Cars, Food-101, Oxford-IIIT Pet
- **`utils/`**:
  - `detection_util.py`: Core scoring functions (`get_ood_scores_clip`, `get_Mahalanobis_score`)
  - `train_eval_util.py`: Dataset loader setup for ID/OOD datasets
  - `args_pool.py`: Argument mappings and configurations
  - `plot_util.py`: Visualization utilities
  - `read_sd_image.py`: Stable diffusion image loading utilities
- **`OODNegMining/`**:
  - Prompt engineering and negative label selection
  - `select_neglabel_cli.py`: CLI script to generate positive/negative text prompts from WordNet for multiple datasets
  - Stores generated prompts in `OODNegMining/<dataset>/positive/` and `OODNegMining/<dataset>/negative/`
- **`scripts/`**: Shell wrappers for common evaluation scenarios

### OOD Scoring Methods

The codebase implements multiple OOD detection scores (see `get_ood_scores_clip` in `utils/detection_util.py:317`):

1. **MCM** (Maximum Concept Matching): Max cosine similarity between image and positive text features
2. **DualCnst**: Combines text-image and image-image similarity using generated images
   - Extracts intermediate layer features from CLIP ViT
   - Caches computed similarities in `similarity_cache/`
3. **NegLabel**: Grouping-based scoring with positive and negative prompts
4. **Energy**: Temperature-scaled logsumexp score
5. **Mahalanobis**: Class-conditional Gaussian modeling with precision matrix

### Key Implementation Details

- **Feature Extraction**: CLIP features are extracted via `net.encode_image()` and normalized
- **Intermediate Features**: For DualCnst scoring, features from specified transformer layers (via `selected_layers`) are pooled and combined with final features
- **Caching**: Generated image features are cached to `similarity_cache/` (controlled by `--cache_path` and `--intermediate_cache_path`)
- **Grouping Strategy**: NegLabel uses 100 groups by default for combining positive/negative similarities (see `grouping()` in `detection_util.py:280`)

### Dataset Structure

Expected directory layout:
```
datasets/
├── ImageNet/
│   ├── train/
│   └── val/
├── ImageNet10/
│   ├── train/
│   └── val/
├── ImageNet20/
│   ├── train/
│   └── val/
├── ImageNet100/
│   ├── train/
│   └── val/
└── ImageNet_OOD_dataset/
    ├── iNaturalist/
    ├── SUN/
    ├── Places/
    └── dtd/
```

Class lists for ImageNet subsets are in `data/<dataset>/class_list.txt`.

### Results Organization

- **`results/<in_dataset>/<score>/<model>_<ckpt>_T_<T>_ID_<name>/`**:
  - Raw results, logs, and evaluation metrics
  - CSV files with human-readable tables (FPR95, AUROC, AUPR) generated by `save_as_dataframe()`
  - Example: `results/ImageNet/DualCnst/CLIP_ViT-B-16_T_0.01_ID_eval_ood/eval_ood.csv`

## Development Workflow

1. **Modifying Prompts**: Edit files in `OODNegMining/<dataset>/` or regenerate with `select_neglabel_cli.py --dataset <dataset_name>`
2. **Adding New Datasets**:
   - Add loader in `utils/train_eval_util.py` (`set_train_loader`, `set_val_loader`, `set_ood_loader_ImageNet`)
   - Update `args_pool.py` for class count mappings
3. **New Scoring Methods**: Extend the score logic in `utils/detection_util.py::get_ood_scores_clip()`
4. **Caching**: Similarity caches are dataset-specific - clear or rename when changing prompts/layers

## Notes

- The repository uses PEP 8 style with 4-space indentation
- Generated images for DualCnst scoring should be placed in the path specified by `--data_dir` (default: `/data0/fayi/generation_image_xl`)
- Keep large datasets/checkpoints outside the repo and use symlinks in `datasets/`
- When modifying intermediate feature extraction layers, clear the intermediate cache files to force recomputation
