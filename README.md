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

This repository provides code and workflows for imputing missing Aerosol Optical Depth (AOD) data using a GAIN-based deep learning model. The imputation framework is designed for satellite-derived AOD products with missing observations and validated using AERONET ground truth data.

## Package Installation

This project requires Python 3.10 or later. Install all required dependencies using the `requirements.yml` file:

```bash
# Create and activate the Conda environment
conda env create -f requirements.yml
conda activate aq-env
```

## Data Preparation
1. **Download the AOD Dataset**
   Download the `.h5` AOD dataset from the following link:
   [Download AOD Dataset](https://gmuedu-my.sharepoint.com/:f:/g/personal/asrireng_gmu_edu/Ei3caNSZZl9Hqq9zkfNDvZMBz3AMiquIi6qvVmeax-TOZg?e=YM6lHR)

2. **Move the File**
   After downloading, place the `.h5` file into the `data/` directory of the project:

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

This work is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)


## Citation

```
```
