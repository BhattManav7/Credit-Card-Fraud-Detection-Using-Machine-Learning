# Credit Card Fraud Detection Using Machine Learning

This project aims to detect fraudulent credit card transactions using classical machine learning algorithms. It leverages a real-world anonymized dataset and compares multiple models to identify the most effective fraud detection strategy.

## Abstract

Credit card fraud leads to billions of dollars in global losses every year. This project proposes a machine learning-based solution to identify fraudulent transactions by learning patterns from historical data. The model was trained and evaluated using the publicly available Kaggle credit card dataset. Algorithms like Logistic Regression, Support Vector Machine, K-Nearest Neighbors, and Decision Tree were explored and compared.

## Project Goals

- Detect fraudulent credit card transactions
- Handle imbalanced data using oversampling techniques (SMOTE)
- Compare performance across multiple machine learning algorithms
- Visualize model performance using evaluation metrics
- Identify the most effective model for fraud detection

## Dataset

- Source: Kaggle - Credit Card Fraud Detection
- Total records: 284,808 transactions
- Fraudulent transactions: 492
- Features:
  - 28 anonymized PCA-transformed variables (V1–V28)
  - Time (seconds since the first transaction)
  - Amount (transaction value)
  - Class (1 = fraud, 0 = not fraud)

## Algorithms Used

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree

## Techniques Applied

- Exploratory Data Analysis (EDA)
- Data standardization
- Handling class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
- Train-test split for model evaluation
- Metrics used: Accuracy, Precision, Recall, F1-Score, ROC–AUC Curve, Confusion Matrix

## Results

- KNN and Decision Tree performed best in terms of accuracy and fraud detection
- SMOTE significantly improved the models’ ability to detect rare fraudulent cases
- Emphasis was placed on recall and precision to minimize false negatives and ensure reliable fraud detection

## Future Work

- Apply ensemble techniques like Random Forest and XGBoost
- Explore real-time fraud detection pipelines
- Integrate additional features such as user location or transaction behavior
- Test with larger and more diverse datasets

## Conclusion

This project successfully demonstrated how machine learning can be applied to credit card fraud detection. Among the tested models, KNN and Decision Tree provided the highest performance. These insights can contribute to building more secure transaction systems and enhancing customer trust.

