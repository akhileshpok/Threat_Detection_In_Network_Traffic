# 🗣️ Threat Detection in Network Traffic

## 🚀  Project Overview
This project is an advanced cybersecurity solution that uses machine learning to protect networks from attacks. The system employs two different models:

- A **binary classifier** that acts as the first line of defense by accurately separating normal network activity from malicious threats.
- A **multi-class classifier** that identifies the specific type of attack (e.g., DoS, PortScan, etc.).

The models are highly effective, demonstrating exceptional performance in distinguishing between benign and malicious traffic. They are designed to minimize false alarms while maintaining a very high rate of successful detection, providing security teams with a reliable and powerful tool to defend against modern cyber threats.


---

## 🎯 Business Goals
- **Risk Mitigation:**  
  The project protects the business from financial loss, data breaches, and service downtime by enabling the early and accurate detection of cyber threats.
- **Operational Efficiency:**  
  The machine learning models reduce false alarms and provide a clear classification of attacks, allowing security teams to respond faster and more effectively.
- **Compliance & Reputation:**  
  By demonstrating a robust cybersecurity posture, the project helps the organization comply with regulations and builds trust with customers and stakeholders.
- **Strategic Advantage:**  
  The insights gained from the models help the business understand its unique threat landscape, enabling proactive security decisions and a more resilient defense against evolving attacks.

---

## 📖 Jupyter Notebooks
Explore the Jupyter notebooks for this project containing the Python code for the ML pipelines, including comments, results, plots, in-depth analysis and insights:

- [Data Prep Script](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/notebooks/Capstone_project_data_prep.ipynb) - This Python script automates the full data preparation and cleaning pipeline for the CICIDS2017 dataset. The code first loads all of the daily CSV files, combines them into a single large DataFrame, and performs initial cleaning steps like standardizing column names and handling infinite or missing values. Finally, it creates two separate, ready-to-use datasets: one for a multiclass classification task that preserves the original attack types, and a second for a binary classification task where all attacks are simplified to a single "malicious" label.
- [BinaryClass ML pipeline](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/notebooks/CICIDS2017_BinaryClass_Pipeline.ipynb) - Notebook with the Python code for the Binary Class ML pipeline. It includes all stages of the pipeline that are described in the **Project Workflow** section. 
- [MultiClass ML pipeline](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/notebooks/CICIDS2017_MultiClass_Pipeline.ipynb) - Notebook with the Python code for the Multi-Class ML pipeline. It includes all stages of the pipeline that are described in the **Project Workflow** section.

---
  

## ⚙️ Project Workflow

### CICIDS2017 Binary Classification, Multi-Class Classicification Pipelines:

This project implements two complete machine learning pipelines to detect network intrusions using the CICIDS2017 dataset. The pipelines are composed of the following stages:

#### 1. Data Loading
- Loads a pre-processed subset of the CICIDS2017 dataset and performs stratified sampling to create a smaller subset for efficient processing.
- Ensures that the label column is binary: `"BENIGN"` or `"ATTACK"` for the binary classifier.
- Ensures that the label column has 5 distinct classes for the multi-class classifier. 
- Uses `LabelEncoder` to convert categorical labels into a numerical format.

#### 2. Exploratory Data Analysis (EDA)
- **Initial Data Inspection**  
  Displays the first few rows of the dataset, data types, non-null counts, and descriptive statistics for numerical features.
- **Data Quality Checks**  
  Identifies and reports any missing values and unique values per feature.
- **Target Variable Analysis**  
  Visualizes the class distribution of the target variable (`label_final_display`) to check for class imbalance.
- **Feature Distributions**  
  Plots histograms and box plots for key numerical features to understand their distributions and identify potential outliers.
- **Relationship Analysis**  
  Generates a correlation heatmap to show the relationships between all numerical features and creates a pair plot to visualize relationships between a selected subset of features, colored by the target class.


#### 3. Train/Test Split
- Splits the dataset into:
  - **Training set**
  - **Validation set**
  - **Test set**
- Stratified sampling is applied to maintain label balance across splits.

#### 4. Feature Engineering
- **Standard Scaling:** Normalizes features using `StandardScaler`.
- **Dimensionality Reduction:** Reduces feature dimensionality with Principal Component Analysis (PCA).
- **Class Imbalance Handling:** SMOTE has been used to synthetically balance the training dataset.

#### 5. Classification Task: ML Modeling
Trains and evaluates the following models:
- **Dummy Classifier:** Serves as a naive baseline for comparison.
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Feedforward Neural Network (FFN)**
- **Long Short-Term Memory (LSTM)**
- **Stacking Classifier:** An ensemble model that combines predictions from Logistic Regression, Random Forest, and XGBoost using a meta-learner for improved performance.

#### 6. Hyperparameter Tuning (Expland)
- **(a)** Binary classification - Uses `GridSearchCV` to tune hyperparameters for Logistic Regression, Random Forest & XGBoost models.
- **(b)** Multi-Class classification - Uses `Optuna` to tune hyperparameters for XGBoost model.
- **(c)** Trains the Stacking Classifier using tuned base learners and a meta-classifier.
- **(d)** Applies the `Optuna` library to tune the Feedforward Neural Network (FFN).
- **(e)** Uses the `Optuna` library to optimize the LSTM model for temporal feature learning.

#### 7. Comparative Performance Evaluation
- Assesses all models on validation and test sets using:
  - Accuracy, Precision, Recall, F1-score (macro & weighted)
  - Confusion Matrix
  - ROC and Precision-Recall curves
- Visualizes metric comparisons using bar charts and performance plots.

#### 8. Model Selection
- Selects the best model based on macro-averaged F1-score.
- Logs the final model name and saves its predictions and probabilities.

#### 9. Model Interpretation
- Uses the `SHAP` library to interpret predictions of the best model.
- Generates:
  - **Global feature importance** via summary plots.
  - **Local explanations** using SHAP force plots.
- Applies interpretation on PCA-transformed features with generic names.

#### 10. Save the Best Model
- Saves the complete pipeline using `joblib` and `numpy`, including:
  - Trained best model
  - `StandardScaler`, `PCA` transformer, and `LabelEncoder`
  - Final predicted labels and probability distributions

#### 11. Implementation (Deployment Readiness)
- Demonstrates how to reload all saved components to make predictions on unseen or real-time data.
- Ensures the pipeline is production-ready, modular, and reproducible.

---

## 📊 CICIDS2017 Dataset Datasheet
The Datasheet for the publicly available CICIDS2017 dataset is available [here](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/data_sheet/CICIDS2017_Dataset_Datasheet.md).

## 🧠 Model Cards
The Model Card for the Binary Clasifier is available [here](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/model_card/model_card_binary.md).

The Model Card for the Multi-class Clasifier is available [here](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/model_card/model_card_multiclass.md).

## 💡 Key Findings
### Binary Classification
- **ROC Curves Comparison:** 
<p align="center">
  <a href="Binary ROC Curves">
    <img src="https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/images/roc_curve_comparison.png" alt="Alt text" width="800" />
  </a>
</p>

- **Precision Recall Curves Comparison:** 
<p align="center">
  <a href="Binary PrecRecall Curves">
    <img src="https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/images/precision_recall_curve_comparison.png" alt="Alt text" width="800" />
  </a>
</p>

- **Top 3 Models:**

<div align="center">

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>F1 (Pos)</th>
      <th>ROC AUC</th>
      <th>Avg Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Stacking</td>
      <td>0.9864</td>
      <td>0.9995</td>
      <td>0.9979</td>
    </tr>
    <tr>
      <td>RF (Tuned)</td>
      <td>0.9862</td>
      <td>0.9991</td>
      <td>0.9973</td>
    </tr>
    <tr>
      <td>XGBoost (Untuned)</td>
      <td>0.9672</td>
      <td>0.9992</td>
      <td>0.9970</td>
    </tr>
  </tbody>
</table>

</div>

The top-performing models (**Stacking, Random Forest, XGBoost**) perform exceptionally well across all key metrics:

- **F1-score (Positive Class):** ~0.986 — very high, indicating excellent precision and recall balance.

- **ROC AUC:** ~0.999 — near-perfect ability to separate classes.

- **Average Precision:** ~0.997 — strong performance even under class imbalance.

The **Deep Learning (DL)** models exhibit decent performance, but not as good as tree-based models. Possible reasons could be:
  - DL models may require more data or more careful tuning (e.g. learning rates, batch sizes).
  - Noisy or redundant features may reduce effectiveness.
  - Overfitting if regularization/dropout wasn’t sufficient.

**Logistic Regression** performs poorly, likely due to linear model’s limitations on non-linear separability, class imbalance and it being weak at capturing complex traffic patterns. 

---

## 🤝 Business Recommendations
To improve Dutch language proficiency among learners, policymakers and educators should consider:


---

## 🔬 Future Work & Enhancements
### Model Optimization
Explore more sophisticated deep learning architectures like CNNs and advanced ensemble techniques to further improve the performance of both the binary and multiclass classifiers.
### Data & Feature Refinement
Develop new, domain-specific features and use advanced feature selection methods to enhance model efficiency and interpretability. Additionally, investigate cost-sensitive learning or other advanced resampling techniques to better handle the dataset's class imbalance.
### Productionization
Focus on building a complete, end-to-end deployment pipeline that includes real-time inference, model monitoring, and automated retraining to move the project from a proof-of-concept to a practical, operational solution.
  

---

## 🛠️ Tech Stack
- **Python & Libraries:**  
Python, pandas, NumPy, scikit-learn, XGBoost, SMOTE, SHAP, joblib, Optuna, PyTorch
- **Machine Learning Models:**  
Logistic Regression, Random Forest Classifier, XGBoost Classifier, Stacking Classifier, Feedforward Neural Networks (FFN), Long Short-Term Memory Networks (LSTM)
- **Model Evaluation Metrics:**  
F1-Score (for Positive Class), ROC AUC, Average Precision, Precision, Recall
- **Model Interpretation & Explainability:**  
SHAP, Feature Importance (for tree-based models), Confusion Matrix
- **Feature Engineering:**  
Handling Missing/Infinite Values, Data Scaling (StandardScaler), Dimensionality Reduction (PCA), One-Hot Encoding
- **Data Visualization:**  
Matplotlib, Seaborn
- **Model Persistence:**  
Joblib, Pickle

---

## 📧 Contact Details
Email: akhilesh.pokhariyal@gmail.com

