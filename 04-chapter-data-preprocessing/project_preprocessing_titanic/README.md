# Predicting Passenger Survival on the Titanic

**Project for Chapter 4: Building Good Training Datasets – Data Preprocessing**

---

## 1. Project Goal

The goal of this project was to build a complete data preprocessing pipeline to clean and prepare the messy, real-world Titanic dataset. After cleaning the data, I trained a model to predict passenger survival and used feature selection techniques to identify the most important factors that contributed to a passenger's chance of survival.

---

## 2. Dataset

I used the classic "Titanic - Machine Learning from Disaster" dataset from Kaggle. This dataset is famous for its data quality issues, including:
*   **Missing Values:** The `Age`, `Cabin`, and `Embarked` columns all had missing data.
*   **Mixed Data Types:** The data included a mix of numerical, categorical, and text features that needed to be handled differently.

---

## 3. My Approach

This project was all about thoughtful data preparation. My workflow included:

1.  **Handling Missing Data:** I used median imputation for the `Age` column (since it was skewed) and dropped the `Cabin` column due to having too many missing values (~77%).
2.  **Feature Engineering:** I extracted passenger titles (like "Mr.", "Miss.", "Mrs.") from the `Name` column to create a new, more useful categorical feature.
3.  **Encoding & Scaling:** I one-hot encoded nominal features like `Sex` and `Embarked` and used a `RobustScaler` on numerical features to handle outliers in the `Fare` column.
4.  **Feature Selection:** I then used two different methods to identify the most predictive features: L1 Regularization (which pushes irrelevant feature weights to zero) and Random Forest Feature Importance.

---

## 4. Results & Conclusion

Both feature selection methods pointed to the same key predictors of survival. The Random Forest model, which is generally more robust, highlighted these top features:

| Feature | Importance Score |
| :--- | :--- |
| `Fare` | 0.23 |
| `Age` | 0.21 |
| `Hon_Mr` | 0.13 |
| `Sex_male` | 0.11 |

**Conclusion:** After applying the full preprocessing pipeline and training a Logistic Regression model on the selected features, I achieved an accuracy of **82.1%**. The analysis clearly shows that being a woman, having a higher-class ticket, and being younger were the strongest indicators of survival on the Titanic.

---

## 5. How to Run

1.  Clone the main repository.
2.  Ensure you have the required libraries installed: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.
3.  Navigate to this project's directory.
4.  Open and run the `ch04_proj.ipynb` Jupyter Notebook. The dataset is included in this directory.