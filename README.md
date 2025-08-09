# Deep Clustering with Convolutional Autoencoders (Reference Project)

This is a cleaned reference project showcasing a multi-step pipeline for deep feature extraction and unsupervised clustering of Acoustic Emission (AE) signals using Convolutional Autoencoders (CAE). It is organized by step prefixes of the original scripts (1_, 2_, 3_, 4_, ...).

## Paper
- Deep Feature Extraction Based on AE Signals for the Characterization of Loaded Carbon Fiber Epoxy and Glass Fiber Epoxy Composites (Applied Sciences, 2022)
  - Link: https://www.mdpi.com/2076-3417/12/4/1867

## Project structure
- scripts/1_prepare: data preparation variants (all scripts starting with `1_`)
- scripts/2_classic_feature_extraction: classic AE feature extraction (`2_`)
- scripts/3_deep_feature_extraction: CAE training and deep feature extraction (`3_`)
- scripts/4_unsupervised_clustering: clustering on features (`4_`)
- scripts/5_visualization: cumulative/diagnostic plots (`5_`)
- scripts/6_feature_importance: feature importance analyses (`6_`)
- scripts/7_export: exporters for merged features (`7_`)
- scripts/8_plotting: parametric plotters (`8_`)
- architectures/: CAE architecture variants
- utils/: small utilities (e.g., TensorFlow environment checks)
- docs/: example visualizations

## Setup
1) Create environment (Python 3.9+ recommended):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2) Install dependencies:
```
pip install -r requirements.txt
```

## Usage
Run scripts step-wise. Each folder contains multiple variants; filenames retain their original prefixes (1_, 2_, 3_, ...).
- Start with a `1_*` script to prepare data
- Proceed with `2_*` for classic features or `3_*` for CAE deep features
- Use `4_*` for clustering and `5_*` for visualization
- Optional: `6_*` for feature importance, `7_*` for export, `8_*` for plotting

Example commands:
```
python scripts/1_prepare/1_Prepare_Data_CAE.py
python scripts/2_classic_feature_extraction/2_Classic_Feature_Extraction.py
python scripts/3_deep_feature_extraction/3_Deep_Feature_Extraction.py
```
Note: Adjust input/output paths in scripts to your local data directories.

## Citation
If you use this project, please cite:

Potočnik, P.; Misson, M.; Šturm, R.; Govekar, E.; Kek, T. (2022). Deep Feature Extraction Based on AE Signals for the Characterization of Loaded Carbon Fiber Epoxy and Glass Fiber Epoxy Composites. Applied Sciences, 12(4), 1867. https://www.mdpi.com/2076-3417/12/4/1867
