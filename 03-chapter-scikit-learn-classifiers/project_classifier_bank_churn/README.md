# Predicting Bank Customer Churn

**Project for Chapter 3: A Tour of Machine Learning Classifiers**

---

## 1. Project Goal

The goal of this project was to predict which customers at a retail bank are most likely to churn (close their accounts). By identifying these at-risk customers, the bank's marketing team can proactively offer incentives to keep them. This project serves as a practical application of several fundamental classification algorithms.

---

## 2. Dataset

I used the "Bank Customer Churn Prediction" dataset from Kaggle. It's a clean, realistic dataset with 10,000 rows of anonymized customer data, including features like credit score, age, tenure, and account balance. The dataset is slightly imbalanced, with about 20% of customers having churned.

---

## 3. My Approach

My process involved a few key stages:

1.  **Baseline Comparison:** I started by training six different classifiers (like Logistic Regression, SVMs, and Random Forest) to see how they performed "out of the box." Since the data is imbalanced, I used the AUC score as my primary metric instead of just accuracy.
2.  **Hyperparameter Tuning:** I then took the top-performing models and experimented with their key hyperparameters (like regularization strength `C` for SVMs and tree depth for Decision Trees) to find the optimal settings.
3.  **Systematic Search:** Finally, I used `GridSearchCV` to perform a more exhaustive search for the best parameters for the two most promising models: Random Forest and Kernel SVM.

---

## 4. Results & Conclusion

After tuning, both the Random Forest and Kernel SVM models showed strong performance, but the **Random Forest model came out slightly ahead** with a final test AUC score of **0.87**.

| Model | Baseline AUC | Tuned AUC |
| :--- | :--- | :--- |
| **Random Forest** | **0.85** | **0.87** |
| Kernel SVM | 0.86 | 0.86 |
| Logistic Regression | 0.77 | 0.77 |

**Conclusion:** I recommend the tuned Random Forest model to the marketing team. It provides the best predictive power for identifying customers at risk of churning. The model's feature importance analysis also revealed that `Age`, `NumOfProducts`, and `Balance` were the most significant predictors of churn.

---

## 5. How to Run

1.  Clone the main repository.
2.  Ensure you have the required libraries installed: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.
3.  Navigate to this project's directory.
4.  Open and run the `ch03_proj.ipynb` Jupyter Notebook. The dataset is included in this directory.