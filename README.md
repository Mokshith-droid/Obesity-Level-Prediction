# Obesity Level Prediction

## Problem
Predict an individual's obesity category from demographic, eating-habit, physical-activity and health-related attributes.

## Dataset
- File: `ObesityDataSet_raw_and_data_sinthetic.csv`
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- Target column: `NObeyesdad`
- Note: the dataset includes synthetically generated records; it is not a clinical diagnostic dataset.

## Algorithms
Decision Tree, Random Forest, Logistic Regression, Gradient Boosting, SVM and MLP Neural Network.

## Method
1. Load and inspect the CSV.
2. Impute missing numeric and categorical values.
3. Scale numeric features and one-hot encode categorical features.
4. Split data into 80% training and 20% testing using stratification.
5. Train and compare six classifiers.
6. Compare accuracy, balanced accuracy, macro precision, macro recall and macro F1.

## Run
```bash
pip install -r requirements.txt
python main.py
```
Put the dataset CSV inside `data/` before running.

## Outputs
Comparison CSV, best trained model and charts are saved in `results/`.

## Results reference
The project report recorded Gradient Boosting as the best model by macro F1 (0.9500) on its reported split. Run this repository version with your dataset to reproduce and regenerate the metrics.

## Limitations
This is an educational prediction experiment, not medical advice or a diagnostic tool.
