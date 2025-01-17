# Click-Through Rate Prediction

This project explores various machine learning approaches to predict click-through rates (CTR) on the Avazu CTR Prediction dataset.  The goal is to accurately predict the probability of a user clicking on an advertisement.

## Dataset

The project uses the Avazu CTR Prediction dataset from Kaggle ([link to dataset](https://www.kaggle.com/c/avazu-ctr-prediction/data)).  Due to the size of the dataset (over 40 million rows), the notebook uses a smaller sample for faster experimentation; however, the full dataset can be easily used by adjusting the `sample_size` parameter in the notebook.


## Approaches

Four different models were implemented and compared, two each with and without feature engineering:

1. **Dummy Classifier (Baseline):** Predicts the majority class (no click) to establish a baseline performance.
2. **Logistic Regression:**  A simple linear model for binary classification.
3. **Decision Tree Classifier:**  A tree-based model that can capture non-linear relationships.
4. **Random Forest Classifier:** An ensemble of decision trees for improved robustness and accuracy.

For each model (excluding the dummy classifier), both an imbalanced version (using the original class distribution) and a balanced version (using RandomOverSampler) were trained and compared. Feature engineering involved generating time-based features from the 'hour' field and applying Target Encoding to all categorical features and then scaling the numerical features.

## Results

The following table summarizes the performance of each model on the test set:


| Model                                    | Accuracy | Recall | Precision | CrossVal_Mean |
|-----------------------------------------|----------|--------|-----------|----------------|
| Dummy classifier                         | 0.83     | NaN    | 0.00      | 0.83           |
| Logistic Regression (Imbalanced)        | 0.84     | 0.62   | 0.21      | 0.84           |
| Logistic Regression (Balanced)           | 0.72     | 0.72   | 0.72      | 0.72           |
| Decision Tree (Imbalanced)              | 0.78     | 0.35   | 0.35      | 0.78           |
| Decision Tree (Balanced)                 | 0.89     | 0.84   | 0.97      | 0.88           |
| Random Forest (Imbalanced)             | 0.82     | 0.47   | 0.27      | 0.82           |
| Random Forest (Balanced)                | 0.91     | 0.86   | 0.97      | 0.89           |
| Logistic Regression (Feature Eng., Bal.) | 0.69     | 0.67   | 0.74      | 0.68           |
| Decision Tree (Feature Eng., Bal.)       | 0.74     | 0.71   | 0.81      | 0.74           |
| Random Forest (Feature Eng., Bal.)      | 0.90     | 0.85   | 0.96      | 0.88           |




## Visualizations

The notebook includes the following visualizations:

* Histograms of numerical features
* Countplot of the 'click' target variable
* Heatmap of the Pearson Correlation matrix of all features
* ROC curves for each model

