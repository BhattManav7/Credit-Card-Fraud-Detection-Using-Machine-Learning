# Credit-Card-Fraud-Detection-Using-Machine-Learning

- **Data Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
---

##  Exploratory Data Analysis (EDA)

- Checked for null values and outliers
- Visualized class imbalance and correlation matrix
- Analyzed transaction amounts, times, and PCA components
- Detected data skew and scaling needs

---

##  Handling Class Imbalance

- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset and generate synthetic examples of the minority class (fraudulent transactions)

---

##  Models Used

| Model                | Status     |
|---------------------|------------|
| Logistic Regression |  Trained |
| Support Vector Machine (SVM) |  Trained |
| K-Nearest Neighbors (KNN) |  Trained |
| Decision Tree Classifier |  Trained |

- Evaluated using **Confusion Matrix**, **Precision**, **Recall**, **F1-score**


---

##  Results & Evaluation

- Focused on **Recall and Precision** due to the importance of minimizing false negatives in fraud detection
- Observed trade-offs between sensitivity and specificity across models
- Final ROC-AUC scores helped identify the best-performing model under imbalanced conditions

---

##  Tech Stack

- **Language**: Python 3.x  
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `SMOTE`

---

##  Key Learnings

- Importance of handling class imbalance in real-world datasets
- Strengths and weaknesses of classic ML algorithms under skewed data
- Role of evaluation metrics beyond accuracy in fraud detection

---

##  Future Work

- Experiment with ensemble methods like Random Forest or XGBoost
- Implement deep learning models (e.g., autoencoders for anomaly detection)
- Real-time fraud detection pipeline simulation



