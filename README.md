# AOD Imputation with GAIN: Reconstructing Satellite AOD Gaps Using Deep Generative Models

## Table of Contents

* [Overview](#overview)
* [Package Installation](#package-installation)
* [Data Preparation](#data-preparation)
* [Running the Imputation Model](#running-the-imputation-model)
* [Validation with AERONET](#validation-with-aeronet)
* [License](#license)
* [Citation](#citation)

## Overview

This repository provides code and workflows for imputing missing AOD (Aerosol Optical Depth) data using a GAIN-based deep learning model. The imputation framework is designed for satellite-derived AOD products with missing observations and validated using AERONET ground truth data.

## Package Installation

This project requires Python 3.10 or later. Install all required dependencies using the `requirements.yml` file:

```bash
# Create and activate the Conda environment
conda env create -f requirements.yml
conda activate gain-aod
```

## Data Preparation

1. Download the `.h5` AOD dataset using the link provided in `data/README.md`.
2. Place the downloaded `.h5` file into the `data/` directory.
3. Run the following script to prepare the training and test data:

```bash
python data_preparation.py
```

This script will extract relevant features and split the data for training and evaluation.

## Running the Imputation Model

To run the GAIN-based AOD imputation model, execute:

```bash
python gain_model.py
```

The model will train on the incomplete AOD dataset and generate reconstructed AOD values.

## Validation with AERONET

To evaluate the model performance against AERONET ground-truth measurements, use the validation script:

```bash
python aeronet_validation.py
```

This will compute RMSE, MAE, and correlation statistics comparing imputed AOD with AERONET measurements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

```
@article{Your2025AOD,
  title     = {Deep Learning-Based Imputation of Satellite AOD Using Generative Models: A Comparative Validation with AERONET},
  author    = {Your Name and Co-authors},
  year      = {2025},
  journal   = {Environmental Data Science (in preparation)}
}
```
