# Evaluating Dimensionality Reduction for Image Classification

**Project for Chapter 5: Compressing Data via Dimensionality Reduction**

---

## 1. Project Goal

This project explores how dimensionality reduction techniques like PCA and LDA can simplify a high-dimensional dataset—in this case, the Fashion-MNIST image dataset—while preserving its predictive power. The main goal was to see if I could achieve similar (or even better!) classification accuracy with fewer features, which would make the models faster and more efficient.

---

## 2. Dataset

I used the popular **Fashion-MNIST dataset** from Kaggle. It's a collection of 70,000 grayscale images of clothing items, each flattened into a 784-pixel vector.

---

## 3. My Approach

I followed a pretty standard workflow for this analysis:

1.  **Preprocessing:** First, I scaled the pixel data to ensure that no single feature dominated the analysis.
2.  **Dimensionality Reduction:** I applied both PCA (unsupervised) and LDA (supervised) to reduce the 784 features down to a much smaller number of components.
3.  **Modeling & Comparison:** I then trained several classifiers (Logistic Regression, Random Forest, and SVC) on both the original and the reduced datasets to compare their performance and speed.

---

## 4. Results & Conclusion

The results were really interesting!

| Model | Data | Accuracy |
| :--- | :--- | :--- |
| Logistic Regression | Original (784 features) | 84.5% |
| **Random Forest** | **PCA (137 features)** | **86.0%** |
| Linear SVC | PCA (137 features) | 84.8% |

**Conclusion:** By reducing the data from 784 features to just 137 using PCA, the **Random Forest Classifier actually performed better** and trained significantly faster.

---

## 5. How to Run

1.  Clone the main repository.
2.  Make sure you have the required libraries installed (Pandas, cuML, Scikit-Learn, etc.).
3.  Navigate to this project's directory.
4.  Open and run the `ch05_proj.ipynb` Jupyter Notebook. The dataset is included in this directory.