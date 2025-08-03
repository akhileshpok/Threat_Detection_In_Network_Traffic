# 🗣️ Threat Detection in Network Traffic

## 🚀  Project Overview
This project is an advanced cybersecurity solution that uses machine learning to protect networks from attacks. The system employs two different models:

- A **binary classifier** that acts as the first line of defense by accurately separating normal network activity from malicious threats.
- A **multiclass classifier** that identifies the specific type of attack (e.g., DoS, PortScan, etc.).

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

### CICIDS2017 Binary Classification Pipeline:

This project implements a complete machine learning pipeline to detect network intrusions using the CICIDS2017 dataset. The pipeline is composed of the following stages:

#### 1. Data Loading
- Loads a pre-processed subset of the CICIDS2017 dataset.
- Ensures that the label column is binary: `"BENIGN"` or `"ATTACK"`.
- Uses `LabelEncoder` to convert categorical labels into numeric format.

#### 2. Exploratory Data Analysis (EDA)
- Examines class distribution to detect imbalance.
- Summarizes dataset statistics (mean, variance, etc.).
- Confirms there are no missing values or NaNs.

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

#### 6. Hyperparameter Tuning
- **(a)** Uses `GridSearchCV` to tune hyperparameters for Logistic Regression, Random Forest, and XGBoost.
- **(b)** Trains the Stacking Classifier using tuned base learners and a meta-classifier.
- **(c)** Applies the `Optuna` library to tune the Feedforward Neural Network (FFN).
- **(d)** Uses the `Optuna` library to optimize the LSTM model for temporal feature learning.

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

## 🧠 Model Card
The Model Card for this project is available [here](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/model_card/model_card.md).

## 💡 Key Findings
- Enrollment in Dutch language courses is the strongest predictor of speaking proficiency. Structured learning is significantly more effective than passive exposure.  
- Native speakers of Germanic languages (e.g., German, English, Swedish) perform best, while non-Germanic Indo-European and non-Indo-European speakers face greater challenges.  
- Monolingual individuals struggle more, whereas those with a second language, especially a Germanic one, have an advantage.  
- Age of arrival matters – Younger arrivals tend to perform slightly better in Dutch proficiency tests.  
- Length of residence alone has a weak effect – Long-term residence only improves proficiency when combined with active learning.  
- General formal education has minimal impact – Learning Dutch through structured courses is more influential than overall schooling background.  

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

