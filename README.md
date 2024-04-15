# Steel Plate Defect Prediction

## Table of Contents

- [Steel Plate Defect Prediction](#steel-plate-defect-prediction)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Data Cleaning and Handling Missing Values](#data-cleaning-and-handling-missing-values)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Pie plot of types of faults](#pie-plot-of-types-of-faults)
    - [Histogram with KDE of all the numerical columns](#histogram-with-kde-of-all-the-numerical-columns)
    - [Correlation matrix](#correlation-matrix)

## Introduction

- **Competition Title**: Steel Plate Defect Prediction
- **Goal**: Predict defects in steel plates using machine learning techniques
- **Ranking**: 70 out of 2199 participants
- **Approach**:
  - Utilized XGBoost, CatBoost, and LightGBM 
  - Ensembled the best performing models.
  - Achieved an accuracy of 0.88905 on the test dataset

## Data Cleaning and Handling Missing Values

- Conducted comprehensive data cleaning to address inconsistencies and errors.
- Implemented custom functions for efficient cleaning of the columns.

## Exploratory Data Analysis (EDA)

Insights were derived through various visualizations:

### Pie plot of types of faults
![Box Plot](./plots/fault.png)

### Histogram with KDE of all the numerical columns
![Pair Plot](./plots/hist_kde_plot.png)

### Correlation matrix
![Pair Plot](./plots/corr.png)
