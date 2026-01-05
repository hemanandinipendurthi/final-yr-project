
# Chronic Heart Disease Risk Classification using Mixed Sensor and Clinical Data

## Project Overview
This project aims to classify patients at different risk levels for **chronic heart disease** using a combination of **sensor data** (ECG, SpO₂, BP) and **clinical categorical features** (chest pain, shortness of breath, etc.). The dataset is **tabular**, synthetic, and designed for **machine learning** and **big data scaling studies**.

### Key Research Questions
1. Can mixed sensor readings and clinical data accurately classify patients at risk for chronic heart disease?
2. Which sensor features are the strongest predictors of heart disease risk?
3. How does machine learning performance change when scaling the dataset?

### Dataset Information
- **Source:** Mendeley Data (Elsevier) — [https://data.mendeley.com/datasets/gsmjh55sfy/1](https://data.mendeley.com/datasets/gsmjh55sfy/1)
- **Type:** Tabular (Numerical + Categorical)
- **Target Variable:** `Triage` — Multiclass (URGENT, SICK, RISK, NORMAL, COLD STATE)
- **Size:** 50,000 records (Big Data)

## Key Steps in the Project

1. **Data Loading and Inspection**: 
   - The dataset is loaded from a CSV file, and basic information such as shape and first few rows are displayed.
   - Missing values and summary statistics are checked to understand the data's structure and potential data quality issues.

2. **Data Cleaning and Preprocessing**: 
   - Label encoding is applied to categorical features to convert them into numerical values for model compatibility.
   - Feature scaling is applied to numerical data for models that are sensitive to feature magnitudes (e.g., Logistic Regression, SVM).

3. **Exploratory Data Analysis (EDA)**:
   - Histograms, boxplots, and correlation heatmaps are generated to explore the distribution of features, detect outliers, and analyze feature relationships.
   - A pair plot and average sensor values by triage category are visualized.

4. **Feature Encoding and Scaling**:
   - Categorical features are label-encoded using **LabelEncoder**.
   - Numerical features are scaled using **StandardScaler** to prepare the data for modeling.

5. **Model Training**:
   - Models like **Logistic Regression**, **Random Forest**, and **Support Vector Machines (SVM)** are trained using the preprocessed data.
   - Performance metrics such as **accuracy**, **F1 score**, **confusion matrix**, and **ROC curve** are calculated for each model.

6. **Model Evaluation**:
   - The models are evaluated using standard classification metrics, and their performances are compared.
   - The final model is selected based on its performance and used for further analysis or deployment.

## Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib** & **seaborn**: Data visualization
- **scikit-learn**: Machine learning, including data preprocessing, model training, and evaluation
- **time**: Measuring the execution time for model training and testing

## How to Run the Project

### 1. Install Required Libraries
Ensure you have all the required libraries by running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Load and Process Data
The data can be loaded from a CSV file using pandas:

```python
df = pd.read_csv("path_to_your_data.csv")
```

### 3. Data Preprocessing
The preprocessing steps include handling missing values, label encoding categorical features, and scaling numerical features.

### 4. Model Training
The model training involves splitting the dataset into training and testing sets and training different machine learning models such as Logistic Regression, Random Forest, and SVM.

### 5. Evaluation
Use accuracy, F1 score, confusion matrix, and ROC curve to evaluate the model performance.

## Results
- The final model's performance metrics are displayed, including confusion matrix and ROC curve for visual analysis.
- The model trained with the best accuracy and F1 score is selected for deployment or further analysis.

## Conclusion
This project successfully demonstrates how **machine learning** can be applied to predict **chronic heart disease** risk levels based on mixed sensor and clinical data. Future work could involve hyperparameter tuning for better model performance or using real-world datasets for validation.
