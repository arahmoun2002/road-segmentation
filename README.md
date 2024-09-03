# Road Segmentation Competition

Welcome to the Road Segmentation Competition project! This repository contains the code and resources for a competition focused on road segmentation in images.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Postprocessing & Evaluation](#postprocessing)

## Introduction
In this competition, participants are challenged to develop an algorithm that can accurately segment roads in satellite images. The goal is to create a model that can identify and outline road areas, which can be used for various applications such as urban planning, traffic analysis, and navigation systems.

## Dataset
To load, augment and prepocess all datasets, call their respective method in `src/preprocessing/preprocess_data.py`.
We used multiple datasets for training our models:
1. [The CIL Road-Segmentation Kaggle competition dataset](https://www.kaggle.com/competitions/ethz-cil-road-segmentation-2024)
    - The respective function to load this dataset is `get_preprocessed_data()`
2. [The DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
    - The respective function to load this dataset is `get_deepglobe_data()`
3. [The Massachusetts Roads Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset)
    - The respective function to load this dataset is `get_deepglobe_data()`
4. [A satellite image dataset sampled from the Google Maps API](https://developers.google.com/maps)
    - To augment the Google image dataset, follow these steps:
      1. Open the `googlemaps.ipynb` notebook located in the `src/preprocessing/data_augmentation` directory.
      2. Run the notebook to download the images and perform data augmentation.
    - The respective function to load this dataset is `get_google_maps_data()` 

## Models
All models are located in the `src/models` directory. 
- To train a model, execute the code-cells in its respective .ipynb notebook. This will also save the model parameters in a directory which you can specify.
- For the ensemble model in `src/models/ensemble.ipynb`, you first need to train all models separately in order to save their parameters.

## Postprocessing & Evaluation
To postprocess the predictions of a model, you can use the `src/postprocessing/postprocessing.ipynb` notebook.
You only need to define the path to the model parameters of the model you want to use in the model loading cell.
