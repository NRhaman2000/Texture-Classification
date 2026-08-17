# Texture Classification: Hand-Crafted Filter Banks vs. Learned Features

Course project for EECS 4422 (Computer Vision). We compare a classical texture
recognition pipeline against a convolutional neural network trained from scratch on
the Describable Textures Dataset (DTD), and additionally evaluate pre-trained models
as an extension beyond the required scope.

## Dataset

DTD contains 47 texture categories with 120 images each, split evenly into train,
validation and test partitions (1,880 images each). All results below use **official
partition 1**. The dataset downloads automatically on first run.

## Results

All models evaluated on the same test split (DTD partition 1, seed 42).

### Required comparison

| Model | Top-1 | Macro-F1 |
| :--- | ---: | ---: |
| Classical: Gabor + Fourier + LBP -> SVM | **31.01%** | **30.70%** |
| CNN trained from scratch (3 seeds) | 18.99% +/- 0.24% | 18.19% +/- 0.31% |

The hand-crafted pipeline outperforms the from-scratch CNN by 12 percentage points.
Analysis of the CNN's first convolutional layer shows its filters remain close to
their random initialization (28.1% relative change; 0.965 cosine similarity to init;
6x less orientation-selective than a Gabor kernel), indicating the network memorized
the training set through its deeper layers rather than learning transferable texture
features. Training loss fell to 0.62 while validation accuracy plateaued near 19%.

### Data efficiency

| Training data | Classical | CNN from scratch |
| :--- | ---: | ---: |
| 25% (470 images) | 21.33% | 11.60% |
| 50% (940 images) | 25.37% | 14.15% |
| 100% (1,880 images) | 31.01% | 18.94% |

The classical pipeline trained on 25% of the data outperforms the CNN trained on all
of it. The gap widens with more data (9.7 -> 12.1 points), so the CNN is not simply
one dataset-size away from catching up.

### Transfer learning (extension, beyond required scope)

| Model | Top-1 |
| :--- | ---: |
| ResNet-50, frozen backbone | ~64.31% |
| EfficientNetV2-S, fine-tuned | ~72.29% |
| ConvNeXt-Tiny, fine-tuned + TTA | ~74.84% |
| ConvNeXt + EfficientNetV2 ensemble + TTA | ~75.90% |

Pre-trained models substantially outperform both approaches above, indicating that on
a dataset this small, the origin of the features matters more than the architecture.

## Repository structure

### Shared

| File | Description |
| :--- | :--- |
| `eval_utils.py` | Seeding, evaluation (top-1 and macro-F1), normalized confusion matrices, filter visualization. |
| `run_experiments.py` | Runs the full experiment suite (seeds, data fractions) and writes `results.json`. Resumable. |
| `plot_results.py` | Generates the data-efficiency and seed-variance figures from `results.json`. |

### Required comparison

| File | Description |
| :--- | :--- |
| `Tex_class_classical.py` | Gabor magnitude-energy + Fourier ring/wedge + LBP features, standardized, RBF SVM with grid search. |
| `Tex_class_cnn.py` | Baseline 2-layer CNN. Overfits; retained to document the tuning progression. |
| `Tex_class_cnn_improved.py` | Over-regularized variant (heavy augmentation, 50% dropout). |
| `Tex_class_cnn_balanced.py` | 3-layer CNN used for the main comparison. Validation checkpointing; saves first-layer filter figure. |

### Transfer learning (extension)

| File | Description |
| :--- | :--- |
| `Tex_class_transfer.py` | ResNet-50 with a frozen backbone and a new linear head. |
| `Tex_class_sota.py` | EfficientNetV2-S, fully fine-tuned with cosine LR schedule. |
| `Tex_class_ultimate.py` | ConvNeXt-Tiny, fine-tuned with test-time augmentation. |
| `Tex_class_ensemble.py` | ConvNeXt-Tiny and EfficientNetV2-S, averaged predictions with TTA. |

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/NRhaman2000/Texture-Classification.git
cd Texture-Classification
pip install -r requirements.txt
```

For GPU training, install PyTorch separately using the selector at
https://pytorch.org/get-started/locally/ before installing the remaining
requirements. The default `pip install torch` provides a CPU-only build.

## Usage

```bash
python Tex_class_classical.py        # classical pipeline (CPU is sufficient)
python Tex_class_cnn_balanced.py     # CNN from scratch (GPU recommended)
python run_experiments.py            # full suite: seeds + data-efficiency curve
python plot_results.py               # figures from results.json
```

`run_experiments.py` caches the extracted classical features to
`features_cache.npz`, so repeated classical experiments take seconds rather than
minutes. It writes each result to `results.json` as it completes and skips finished
work if re-run.

## Generated outputs

Running the scripts produces `figures/gabor_bank.png` (the hand-designed filters),
`figures/cnn_conv1_filters.png` (the filters the CNN learned), row-normalized confusion
matrices for each model, and, via `plot_results.py`, `data_efficiency.png` and
`seed_variance.png`.

## Notes on methodology

- All models use DTD partition 1 and are evaluated on the same held-out test split.
- The CNN uses the validation split for epoch selection; the test set is used once.
- CNN results are reported as mean +/- standard deviation over three seeds.
- The classical pipeline is deterministic, so it is reported from a single run.
- The classical pipeline operates on grayscale images while the CNNs receive RGB;
  the CNN therefore has access to color information the classical model does not.
