# Telecom Customer Churn Prediction

## Overview
This project is a comprehensive **Data Science portfolio** piece focused on predicting telecom customer churn through advanced **machine learning** methods. The objective is to develop robust, scalable, and interpretable predictive models to proactively identify at-risk customers, enhancing customer retention strategies.  

The analysis builds upon the foundational approach presented by **Lalwani et al. (2022)**, addressing the challenges of imbalanced datasets in churn prediction using advanced **resampling techniques** (including **undersampling** and **oversampling with SMOTE**) and enhanced evaluation metrics. Key methodological highlights include rigorous **feature selection** using the **Gravitational Search Algorithm (GSA)**, extensive hyperparameter tuning to optimize model performance, and comprehensive comparative analysis across various machine learning models and dataset variations.


## 📂 Project Structure

The project is organized into two main parts:

### Part I: Data Exploration & Preprocessing
1. **Introduction**: Objectives and scope of churn prediction.
2. **Dataset Acquisition & Structure**: Data sourcing and preliminary assessment.
3. **Exploratory Data Analysis (EDA)**: Visualization and feature distribution analysis.
4. **Data Preprocessing**: Handling missing data, transforming categorical features, and data cleaning.

### Part II: Modeling & Evaluation
1. **Feature Selection & Preprocessing**
   - Feature selection using **Gravitational Search Algorithm (GSA)** and statistical methods.
   - Data balancing with **undersampling and SMOTE**.

2. **Model Training & Hyperparameter Tuning**
   - Training supervised learning algorithms (**Logistic Regression, Decision Tree, KNN, Random Forest, Naïve Bayes, XGBoost, CatBoost**).
   - Hyperparameter optimization for improved predictions.

3. **Testing & Model Comparison**
   - Evaluation of predictive performance using multiple metrics (**Accuracy, Precision, Recall, F1-score, AUC-ROC**).
   - Comparing model effectiveness across different feature sets and data balancing methods.

4. **Final Recommendation**
   - Selection of the best-performing, most interpretable predictive model.
   - Insights into how feature selection and balancing methods affect performance.

## ⚖ Model and Dataset Comparison

### Dataset Variants Evaluated:
- **df_original**: Complete dataset with all initial features.
- **df_gsa**: Features selected through **GSA** optimization.
- **df_selected**: Features chosen by multiple methods (**Correlation, Mutual Information, t-test, Random Forest, XGBoost**).
- **df_selected_multi**: Dataset derived from **df_selected** after removing highly correlated features.

### 📈 Key Insights:
| Dataset Variant      | Number of Features | Performance | Interpretability | Recommended |
|----------------------|--------------------|-------------|------------------|-------------|
| **df_original**      | 31                 | Moderate    | Low              | ❌          |
| **df_gsa**           | 17                 | Moderate    | Moderate         | ⚠️          |
| **df_selected**      | 24                 | High        | Good             | ✅          |
| **df_selected_multi**| 19                 | High        | Very High        | ⭐          |

**Recommendation**: The dataset **df_selected_multi** offers the optimal balance of predictive accuracy, computational efficiency, and interpretability, making it ideal for actionable churn prediction strategies.

## 🤖 Machine Learning Models Implemented
- Logistic Regression
- Decision Tree (CART)
- K-Nearest Neighbors (KNN)
- Random Forest
- Naïve Bayes (Gaussian)
- XGBoost
- CatBoost

## 📊 Evaluation Metrics
The following performance metrics were used to evaluate the models:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- AUC-ROC
- False Negative Rate (Type I Error)
- False Positive Rate (Type II Error)

## 🛠 Technologies & Tools Used
- **Programming Language**: Python (Pandas, NumPy, Scikit-learn, XGBoost, CatBoost)
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning Tools**: GridSearchCV, Stratified K-Fold Cross-Validation
- **Data Preprocessing**: SMOTE, StandardScaler, Label Encoding, One-Hot Encoding, Gravitational Search Algorithm (GSA)

## 📂 Dataset Information
- **Source**: [Telco Customer Churn Dataset by Blastchar (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Number of Samples**: 7,043
- **Features**: Customer demographics, account details, service subscriptions, billing information.
- **Target Variable**: Churn (**Yes** or **No**)

## 📅 Conclusion
Accurately predicting customer churn is crucial for telecom providers aiming for proactive retention. This project emphasizes the importance of **feature selection** and proper handling of **imbalanced datasets** to achieve highly effective churn prediction. The chosen model provides robust predictive performance alongside meaningful interpretability, enabling efficient, targeted customer retention initiatives.

## 📚 References
- Lalwani, P., Mishra, M. K., Chadha, J. S., & Sethi, P. (2022). Customer churn prediction system: A machine learning approach. *Computing, 104*, 1497–1516. [doi.org/10.1007/s00607-021-00908-y](https://doi.org/10.1007/s00607-021-00908-y)
- Kaggle - Telco Customer Churn Dataset by Blastchar. [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## ⚠️ **Important Note**  
This project was developed as part of a **Data Science portfolio** to explore the application of **machine learning techniques** for telecom customer churn prediction. The results are **not intended to be used as definitive business strategies or a formal scientific study**, but rather as a demonstration of data-driven approaches to customer retention.

