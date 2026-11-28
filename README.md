# Brain Tumor Detection & Localization

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-orange.svg)
![Keras](https://img.shields.io/badge/Keras-API-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A two-stage Keras pipeline that **detects** and **localizes** lower-grade glioma
on brain MRI slices: a **ResNet50** classifier decides whether a slice shows a
tumor, and a **ResUNet** segmentation network outlines the tumor at the pixel
level for every slice flagged as positive.

This is a course project. It follows the brain tumor case study ("AI in
Healthcare") by Dr. Ryan Ahmed in the Udemy course *Modern Artificial
Intelligence Masterclass: Build 6 Projects*. The notebook does not train either
model: its evaluation cells load pre-trained weights that came with the course.
See [Results](#results) for what that means for the numbers.

<p align="center">
  <b>MRI scan → [ ResNet50: tumor? ] → if yes → [ ResUNet: where? ] → pixel-level mask</b>
</p>

<p align="center">
  <img src="assets/segmentation_results.png" alt="Sample output: MRI, ground-truth mask, predicted mask, and overlays" width="100%">
  <br>
  <em>Output of the course's pre-trained weights on three test slices: MRI · ground-truth mask · predicted mask · ground-truth overlay (red) · predicted overlay (green).</em>
</p>

> **Disclaimer:** This is a research / portfolio project. It is **not** a medical
> device and must not be used for actual diagnosis.

---

## Table of Contents

- [Why two models?](#why-two-models)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [How the code is organized](#how-the-code-is-organized)
- [Tech stack](#tech-stack)
- [Future work](#future-work)
- [License & acknowledgements](#license--acknowledgements)

---

## Why two models?

Every scan raises two clinically different questions — *is there a tumor?* and
*where is it?* — and no single model answers both cleanly:

- A **classifier** alone tells you a tumor exists but not its location.
- A **segmentation** model alone is heavier, harder to train on imbalanced data,
  and wastes compute on the ~65% of slices that show no tumor.

So the pipeline uses a **two-stage cascade**: a classifier screens *every*
slice, and only the positives are handed to the segmentation network.

## Architecture

| Stage | Model | Task | Key idea |
| :---: | --- | --- | --- |
| **1** | ResNet50 (transfer learning) | Binary classification — tumor vs. no tumor | Freeze an ImageNet-pretrained backbone, train only a custom softmax head |
| **2** | ResUNet + Focal Tversky loss | Semantic segmentation — pixel-level mask | U-Net encoder/decoder built from residual blocks + skip connections |

The notebook's freeze loop sets `layers.trainable` (the Keras module) instead of
`layer.trainable`, so it freezes nothing; [`models.py`](models.py) freezes the
backbone correctly. Neither version produced the weights evaluated below.

**ResUNet at a glance:**

```
Encoder (downsampling)                          Decoder (upsampling)
  Stage 1  Conv x2   (16)  ------ skip ------>  concat -> ResBlock (16)
  Stage 2  ResBlock  (32)  ------ skip ------>  concat -> ResBlock (32)
  Stage 3  ResBlock  (64)  ------ skip ------>  concat -> ResBlock (64)
  Stage 4  ResBlock (128)  ------ skip ------>  concat -> ResBlock (128)
              \                                        /
               ------>  Bottleneck ResBlock (256)  ----
  Final: Conv 1x1 + sigmoid  ->  tumor mask (256 x 256 x 1)
```

The segmentation model is compiled with the **Focal Tversky loss**, which weights
missed tumor pixels more heavily than false alarms, because tumor pixels are a
tiny fraction of each slice. See
[`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) for the full technical write-up.

## Dataset

[**LGG MRI Segmentation**](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)
— 2D FLAIR MRI slices of lower-grade glioma patients from The Cancer Genome Atlas
(TCGA), stored in one folder per patient, each slice paired with a manually
annotated binary tumor mask. Every patient has a tumor, so a tumor-negative
slice is one where the tumor is not visible, not a healthy scan.

| Property | Value |
| --- | --- |
| Total slices | 3,929 |
| Tumor-negative (`mask = 0`) | 2,556 |
| Tumor-positive (`mask = 1`) | 1,373 |
| Image size | 256 × 256 × 3 |
| Mask values | 0 (background) / 255 (tumor) |

Two metadata files (`data.csv`, `data_mask.csv`) map each scan to its mask and
store the binary tumor flag.

<p align="center">
  <img src="assets/dataset_samples.png" alt="Sample MRI scans with tumor masks" width="90%">
  <br>
  <em>Tumor-positive samples: MRI · annotated mask · mask overlaid on the scan (red).</em>
</p>

## Results

These numbers are not from a training run in this repo. The notebook's `fit`
cells are commented out, and its evaluation cells load the course's pre-trained
files (`resnet-50-MRI.json` + `weights.hdf5` for the classifier,
`ResUNet-MRI.json` + `weights_seg.hdf5` for the segmenter). What follows is how
those weights score on the notebook's test split.

The ResNet50 classifier was evaluated on **576 test slices** (the split has 590;
`steps = n // 16` drops the last 14):

| Metric | Class 0 (no tumor) | Class 1 (tumor) |
| --- | :---: | :---: |
| Precision | 0.98 | 0.98 |
| Recall | 0.99 | 0.96 |
| F1-score | 0.99 | 0.97 |

Accuracy: 98.1% (565 of 576; 3 false positives, 8 false negatives).

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Classifier confusion matrix on the test set" width="360">
  <br>
  <em>Confusion matrix on the 576-slice test set (rows = actual, cols = predicted).</em>
</p>

Treat the 98.1% as optimistic:

- The test split is random over slices (`train_test_split` with no
  `random_state` and no grouping by patient), so neighbouring slices of the same
  patient can sit on both sides of it.
- The weights were trained on a split this repo can't see, so some test slices
  may have been in their training data.

There is no segmentation metric (Dice or IoU) on held-out data. The ResUNet
results are qualitative only: the overlays at the top of this README and at the
end of the notebook.

## Repository structure

```
.
├── Brain_Tumor.ipynb        # End-to-end pipeline: EDA → classifier → segmenter → evaluation
├── models.py                # Model builders: ResNet50 classifier + ResUNet
├── train.py                 # Trains both stages, writes the weight + JSON files (no results published yet)
├── predict.py               # CLI: run the trained pipeline on new MRI scans
├── utilities.py             # Custom DataGenerator, Focal Tversky loss, two-stage inference
├── requirements.txt         # Pinned Python dependencies
├── docs/
│   └── PROJECT_REPORT.md    # Technical write-up (problem, approach, results, limitations)
├── assets/                  # Result figures used in the README / report
├── LICENSE                  # MIT
├── .gitignore
└── README.md
```

**Not committed** (large, or course material — see `.gitignore`): the MRI
dataset (`Brain_MRI/`, `data.csv`, `data_mask.csv`), the weights
(`weights.hdf5`, `weights_seg.hdf5`) and the serialized architectures
(`resnet-50-MRI.json`, `ResUNet-MRI.json`). [`train.py`](train.py) writes files
with the same names from your own training run — see
[Getting the trained weights](#getting-the-trained-weights) below.

## Getting started

### Option A — Google Colab (recommended)

The notebook was built on Colab with a free GPU. Just open it and run:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VinayakMokashi/brain-tumor-detection-localization/blob/main/Brain_Tumor.ipynb)

1. Open the badge above (Runtime → Change runtime type → **GPU**).
2. Download the [LGG MRI dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)
   and place it in your Google Drive (e.g. `MyDrive/Brain_MRI/`).
3. Update the dataset path cell (`%cd /content/drive/MyDrive/Brain_MRI`) to match
   your folder, then run the cells top to bottom.

### Option B — Run locally

```bash
# 1. Clone
git clone https://github.com/VinayakMokashi/brain-tumor-detection-localization.git
cd brain-tumor-detection-localization

# 2. Create an environment and install dependencies
python -m venv .venv
# Windows:  .venv\Scripts\activate    |    macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook Brain_Tumor.ipynb
```

When running locally, remove the `from google.colab import ...` /
`drive.mount(...)` cells and point the dataset path at your local dataset folder.
A GPU is strongly recommended for training (inference on trained weights runs
fine on CPU, just slower).

### Getting the trained weights

The notebook's evaluation sections load four files that are **not** committed:
`weights.hdf5` + `resnet-50-MRI.json` (classifier) and
`weights_seg.hdf5` + `ResUNet-MRI.json` (segmenter). You have two options.

**1. Train them yourself.** [`train.py`](train.py) builds, trains and saves all
four files, after which the notebook's evaluation cells can load them. No
results from it are published yet. The notebook draws its own unseeded test
split, which will overlap the slices `train.py` trained on, so a fair comparison
needs one fixed, patient-level split for both:

```bash
# trains both stages and writes the .hdf5 / .json files into the data folder
python train.py --data-dir /path/to/Brain_MRI

# handy flags:
#   --stage classifier|segmenter|all   pick a stage
#   --clf-epochs N  --seg-epochs N      epochs per stage
#   --batch-size N                      batch size (default 16)
```

**2. Use pre-trained weights.** The results above come from the pre-trained
files that came with the course; they are not redistributed here. If you have
them, or weights you trained and host yourself (for example on a
[GitHub Release](https://github.com/VinayakMokashi/brain-tumor-detection-localization/releases)),
drop the four files into the dataset folder and run the evaluation cells.

### Run inference on new scans

Once you have the trained weights, [`predict.py`](predict.py) runs the full
pipeline on unseen scans from the command line:

```bash
# single scan
python predict.py --image path/to/scan.tif

# a folder of scans -> overlay PNGs saved in ./predictions
python predict.py --dir path/to/scans --out-dir predictions
```

For each scan it prints whether a tumor was detected and, for positives, saves a
side-by-side **MRI / predicted mask / overlay** figure (tumor region in green).

## How the code is organized

Most of the pipeline lives in the notebook, but the pieces that don't fit
off-the-shelf Keras are factored into [`utilities.py`](utilities.py):

- **`DataGenerator`** — a Keras `Sequence` that streams (MRI, mask) pairs in
  batches, resizing and normalizing on the fly so the full dataset never has to
  sit in memory.
- **`tversky` / `tversky_loss` / `focal_tversky`** — the segmentation
  metric and loss functions (see [Abraham & Khan, 2018](https://arxiv.org/abs/1810.07842)).
- **`prediction`** — runs the complete classify-then-segment inference over a
  test set and returns per-scan masks and a has-tumor flag.

The network architectures live in [`models.py`](models.py) (`build_classifier`,
`build_resunet`), [`train.py`](train.py) wires the data, models and loss together
into a single training run, and [`predict.py`](predict.py) loads the
trained models to run the pipeline on new scans.

## Tech stack

`Python` · `TensorFlow / Keras` · `NumPy` · `pandas` · `OpenCV` ·
`scikit-image` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Plotly`

## Future work

- Train both stages here on a patient-level split (all slices of a patient in
  one split) and report Dice/IoU for the segmenter on held-out patients
- k-fold cross-validation for tighter confidence intervals on the metrics
- Hyperparameter tuning and test-time augmentation
- Uncertainty estimation on the predicted masks
- A small **Streamlit / Flask** web app for interactive inference

## License & acknowledgements

Released under the [MIT License](LICENSE).

- Course: the pipeline, most of the notebook and `utilities.py` code, and the
  pre-trained weights the notebook evaluates come from the brain tumor case
  study ("AI in Healthcare") by Dr. Ryan Ahmed in the Udemy course *Modern
  Artificial Intelligence Masterclass: Build 6 Projects*. `models.py`,
  `train.py`, `predict.py` and the docs were added in this repo.
- Dataset: [LGG MRI Segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) (M. Buda et al.)
- ResNet: He et al., *Deep Residual Learning for Image Recognition* (2015)
- U-Net: Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)
- Focal Tversky loss: Abraham & Khan (2018)

---

*Built by [Vinayak Mokashi](https://github.com/VinayakMokashi).*
