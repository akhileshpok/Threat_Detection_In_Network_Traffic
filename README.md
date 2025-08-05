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

#### 6. Hyperparameter Tuning 
- **(a)** Binary classification - Uses `GridSearchCV` to tune hyperparameters for the Logistic Regression, Random Forest & XGBoost models.
- **(b)** Multi-Class classification - Uses `Optuna` to tune hyperparameters for the XGBoost model.
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

---

## 🧠 Model Cards
- The Model Card for the Binary Clasifier is available [here](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/model_card/model_card_binary.md).

- The Model Card for the Multi-Class Clasifier is available [here](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/model_card/model_card_multiclass.md).

---

## 🎛️ Hyperparameter Optimisation
### Binary Classifier

For the selected binary classifier, which is a **Stacking Ensemble**, the hyperparameter optimisation focused on tuning the individual base learners as well as the meta-learner to achieve the best predictive performance.

#### Base Learners and Hyperparameters Tuned

**Logistic Regression:**
The Logistic Regression base model was tuned using `GridSearchCV` with a pipeline.

- Regularization strength inverse, tested across `[0.01, 0.1, 1, 10, 100]` to control overfitting (`C`)  
- Optimization algorithms tested were `'lbfgs'` (supports only `'l2'` penalty) and `'saga'` (supports `'l1'` and `'elasticnet'` penalties, but here only `'l2'` was used) (`solver`)  
- Only `'l2'` penalty was considered to ensure compatibility with solvers (`penalty`) 


**Random Forest (RF):**
The Random Forest base model was tuned using `GridSearchCV` with a pipeline.

- Number of trees, tested at 100, 200, and 300 (`n_estimators`)  
- Maximum depth of trees, tested at 10, 20, and unlimited (`max_depth`)  
- Minimum samples required to split a node, tested at 2 and 5 (`min_samples_split`)  
- Minium samples required at a leaf node, tested at 1 and 2 (`min_samples_leaf`)
- Number of features considered for the best split, tested with `'sqrt'` and `'log2'` (`max_features`)  

**XGBoost:**
The XGBoost base model was tuned using `GridSearchCV` with a pipeline. 

- Number of boosting rounds, tested at 100 and 200 (`n_estimators`)  
- Maximum depth of trees, tested at 3 and 5 (`max_depth`)  
- Step size shrinkage used to prevent overfitting, tested at 0.01 and 0.1 (`learning_rate`)
- Subsample ratio of the training instances, tested at 0.7 and 1.0 (`subsample`)   

#### Meta-Learner
The meta-learner in the stacking ensemble is a **Logistic Regression** model configured with default parameters optimized for binary classification. Unlike the base learners, explicit hyperparameter tuning was minimal or not performed for the meta-learner due to:
- The meta-learner’s relatively simple architecture requiring fewer hyperparameters.
- Use of the `'lbfgs'` solver with L2 regularization by default, which generally provides robust performance.
- The focus on tuning base learners where most performance gains are realized.

Cross-validation during stacking training (3-fold stratified K-Fold) implicitly ensures the meta-learner generalizes well without overfitting. Future improvements may explore tuning regularization strength (`C`) or solver types if needed.

#### Optimisation Strategy

- **Grid Search / Random Search:** Hyperparameter combinations were explored using Grid Search or Randomized Search with cross-validation on the training set.  
- **Evaluation Metric:** The F1-score for the positive (attack) class was used as the primary metric to balance precision and recall.  
- **Early Stopping:** For models supporting it (e.g., XGBoost), early stopping was used to prevent overfitting.  
- **Resource Considerations:** To manage computational cost, the search space was carefully selected based on domain knowledge and prior experiments.  

---
## 💡 Key Findings
### Key Results and Findings for the Binary Classifier
- **ROC Curves Comparison:** 
<p align="center">
  <a href="Binary ROC Curves">
    <img src="https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/images/roc_curve_comparison.png" alt="ROC Curves Comparison" width="800" />
  </a>
</p>

- **Precision Recall Curves Comparison:** 
<p align="center">
  <a href="Binary PrecRecall Curves">
    <img src="https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/images/precision_recall_curve_comparison.png" alt="Prec-Recall Curves Comparison" width="800" />
  </a>
</p>

- **Top 3 Models (Binary Class):**

<div align="center">

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>F1 (Pos)</th>
      <th>ROC AUC</th>
      <th>Avg. Precision</th>
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

The **Deep Learning (DL)** models show decent results but lag behind tree-based methods, possibly due to:

- Need for more data or refined hyperparameter tuning (learning rate, batch size, etc.).
- Impact of noisy or redundant features.
- Potential overfitting from insufficient regularization.

**Logistic Regression** performs poorly, likely because:

- Linear assumptions limit ability to model non-linear patterns.
- Class imbalance affects performance.
- It’s less capable of capturing complex traffic behaviors.

Overall, the **Stacking Ensemble** offers the best combination of metrics and is the preferred model for deployment in the **binary classification** task.

### Key Results and Findings for the Multi-Class Classifier
- **ROC Curves Comparison:** 
<p align="center">
  <a href="Multi ROC Curves">
    <img src="https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/images/multi_roc_curve_comparison.png" alt="ROC Curves Comparison" width="800" />
  </a>
</p>

- **Precision Recall Curves Comparison:** 
<p align="center">
  <a href="Multi PrecRecall Curves">
    <img src="https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/images/multi_precision_recall_curve_comparison.png" alt="Prec-Recall Curves Comparison" width="800" />
  </a>
</p>

- **Top 3 Models (Multi-Class):**

<div align="center">

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Weighted F1-score</th>
      <th>Micro-average ROC AUC</th>
      <th>Micro-average Precision-Recall AP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Stacking Classifier</td>
      <td>0.9837</td>
      <td>0.9989</td>
      <td>0.9957</td>
    </tr>
    <tr>
      <td>Random Forest (Untuned)</td>
      <td>0.9835</td>
      <td>0.9976</td>
      <td>0.9929</td>
    </tr>
    <tr>
      <td>Random Forest (Tuned)</td>
      <td>0.9819</td>
      <td>0.9988</td>
      <td>0.9960</td>
    </tr>
  </tbody>
</table>

</div>

The top-performing models — **Stacking Classifier**, **Random Forest (Untuned and Tuned)** — demonstrate strong and consistent performance across all key evaluation metrics:

- **Weighted F1-score**: Around **0.98–0.98+**, indicating a well-balanced precision and recall across all classes, including minority ones.
- **Micro-average ROC AUC**: Approximately **0.999**, showing near-perfect discrimination capability across all classes.
- **Micro-average Precision-Recall Average Precision (AP)**: Around **0.995+**, indicating excellent predictive performance even with class imbalance.

**Tree-based models (Random Forest and XGBoost)** outperform deep learning approaches (FFN, LSTM), likely due to:

- Limited data size restricting deep model generalization.
- Deep models needing further tuning and potentially more regularization.
- Noisy or redundant features that tree-based methods handle better.

**Deep Learning models** show reasonable but lower performance, possibly due to:

- Need for more data or better hyperparameter optimization.
- Challenges in capturing complex patterns in multi-class network traffic data.

**Logistic Regression** performs the weakest among tested models, likely because:

- Its linear nature limits capturing non-linear and complex attack patterns.
- Class imbalance affects its sensitivity to minority classes.

**Overall**, the **Stacking Ensemble** provides the best combination of robustness and accuracy for **multi-class classification** on the CICIDS2017 subset and is recommended as the preferred model for deployment.

---

## 🤝 Business Recommendations for Binary Threat Detection (Benign vs. Malicious)

- ✅ **Use the model to detect whether network traffic is safe or potentially malicious**, providing a first layer of defense in your security systems.  
- ✅ **Deploy it in real-time environments** like firewalls, intrusion detection systems (IDS), or SIEM tools to flag suspicious activity early.  
- ✅ **Take advantage of the model’s strong accuracy** to reduce false alarms and avoid missing real attacks.  
- ✅ **Integrate into your security team’s daily workflows** to help analysts quickly filter out safe traffic and focus on real threats.  
- ✅ **Test the model on your own network data** to ensure it works well with your traffic patterns before going live.  
- ✅ **Retrain the model regularly** using new attack data to stay ahead of evolving threats.  
- ✅ **Track the model’s performance over time**, and update it if you notice a drop in accuracy or an increase in false alerts.  
- ❌ **Don’t rely solely on this model to make automated security decisions**—always keep a human in the loop for critical actions.  

## 🤝 Business Recommendations for Multi-Class Attack Detection

- ✅ **Use the model to identify different types of network attacks**, helping security teams respond more effectively to each threat.  
- ✅ **Tailor responses based on the type of attack detected**, such as using specific protections for DDoS attacks versus PortScan attempts.  
- ✅ **Rely on the model’s strong accuracy for common attacks**, but be cautious with rare attack types since they have less data.  
- ✅ **Integrate the model into your existing security processes** to help analysts quickly understand and prioritize threats.  
- ✅ **Test the model on your own network data** before fully relying on it, to make sure it works well in your environment.  
- ✅ **Keep the model up-to-date by retraining it regularly**, so it stays effective against new and changing cyber threats.  
- ✅ **Monitor the model’s performance over time**, and update it if you notice it becoming less accurate.  
- ✅ **Combine the model with other security tools**, like anomaly detection, to catch unusual or new types of attacks.  
- ❌ **Don’t rely only on the model for automatic blocking of network traffic**, especially for rare or high-risk attacks, to avoid mistakes that could disrupt your business.  

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

