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
Explore the Jupyter notebooks accompanying this project, which include Python code for the machine learning pipelines, along with detailed comments, result visualizations, and key insights. 

- [Data Prep Script](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/notebooks/Capstone_project_data_prep.ipynb) - This Python script automates the full data preparation and cleaning pipeline for the CICIDS2017 dataset. The code first loads all of the daily CSV files, combines them into a single large DataFrame, and performs initial cleaning steps like standardizing column names and handling infinite or missing values. Finally, it creates two separate, ready-to-use datasets: one for a multiclass classification task that preserves the original attack types, and a second for a binary classification task where all attacks are simplified to a single "malicious" label.
- [BinaryClass ML pipeline](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/notebooks/CICIDS2017_BinaryClass_Pipeline.ipynb) - Notebook with the Python code for the Binary Class ML pipeline. It includes all stages of the pipeline that are described in the **Project Workflow** section. 
- [MultiClass ML pipeline](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/notebooks/CICIDS2017_MultiClass_Pipeline.ipynb) - Notebook with the Python code for the Multi-Class ML pipeline. It includes all stages of the pipeline that are described in the **Project Workflow** section.

---
  
## ⚙️ Project Workflow

### Binary Classification & Multi-Class Classicification Pipelines:

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
  - **Test set**
- Stratified sampling is applied to maintain label balance across splits.

#### 4. Feature Engineering
- **Standard Scaling:** Normalizes features using `StandardScaler`.
- **Dimensionality Reduction:** Reduces feature dimensionality with Principal Component Analysis (PCA).
- **Class Imbalance Handling:** SMOTE has been used to synthetically balance the training dataset.


#### 5. Classification Task ML Modeling
Trains and evaluates the following models for both **Binary** and **Multi-Class** classification tasks using the CICIDS2017 dataset. The model selection was designed to cover a spectrum from simple to complex, ensuring robust baselines and high-performing classifiers suitable for network intrusion detection.

#### ✅ Models Selected and Justification
- **Dummy Classifier**  
  *Purpose:* Serves as a naive baseline (e.g., always predicting the most frequent class).  
  *Why Chosen:* Establishes a performance floor, allowing meaningful evaluation of more sophisticated models.
- **Logistic Regression**  
  *Purpose:* A linear classifier optimized using gradient descent.  
  *Why Chosen:* Simple, interpretable, and often surprisingly competitive on high-dimensional tabular data. Also used as the meta-learner in stacking due to its generalization ability.
- **Random Forest**  
  *Purpose:* An ensemble of decision trees trained via bootstrap aggregation (bagging).  
  *Why Chosen:* Handles both numerical and categorical features well, robust to overfitting, and provides feature importance. Effective on tabular data with class imbalance.
- **XGBoost**  
  *Purpose:* A gradient boosting framework known for speed and performance.  
  *Why Chosen:* Handles class imbalance, supports regularization, and captures complex interactions. Performs particularly well on structured datasets like CICIDS2017.
- **Feedforward Neural Network (FFN)**  
  *Purpose:* A deep learning model with dense, fully connected layers.  
  *Why Chosen:* Can model complex, non-linear feature interactions. Suitable when the dataset is large and diverse enough to benefit from deep architectures.
- **Long Short-Term Memory (LSTM)**  
  *Purpose:* A type of recurrent neural network (RNN) designed to model sequential or temporal dependencies.  
  *Why Chosen:* Especially useful when modeling time-based or sequential network traffic flows, as LSTM captures dependencies across time steps.
- **Stacking Classifier (Ensemble)**  
  *Purpose:* An ensemble method that combines predictions of multiple base models via a meta-learner.  
  *Why Chosen:* Leverages the strengths of diverse models (e.g., LogReg, RF, XGBoost) to improve predictive performance and robustness. Often outperforms individual classifiers in complex tasks like intrusion detection.

#### 🔍 Summary of Model Selection Strategy

- **Range of Complexity**: From simple baselines (Dummy) to complex deep models (FFN, LSTM).
- **Interpretability vs. Accuracy**: Balances transparent models like Logistic Regression with high-performing black-box models.
- **Robustness**: Ensemble methods (Random Forest, XGBoost, Stacking) increase stability and accuracy.
- **Domain Suitability**: All models are appropriate for structured cybersecurity data with both categorical and numerical features.


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

For the selected binary classifier, which is a **Stacking Ensemble**, the hyperparameter optimisation focused on tuning the individual base learners—including classical machine learning models and deep learning architectures—as well as the meta-learner to achieve the best predictive performance.

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
- Minimum samples required at a leaf node, tested at 1 and 2 (`min_samples_leaf`)  
- Number of features considered for the best split, tested with `'sqrt'` and `'log2'` (`max_features`)  

**XGBoost:**  
The XGBoost base model was tuned using `GridSearchCV` with a pipeline.

- Number of boosting rounds, tested at 100 and 200 (`n_estimators`)  
- Maximum depth of trees, tested at 3 and 5 (`max_depth`)  
- Step size shrinkage used to prevent overfitting, tested at 0.01 and 0.1 (`learning_rate`)  
- Subsample ratio of the training instances, tested at 0.7 and 1.0 (`subsample`)  

**Feedforward Neural Network (FFN):**  
The FFN model was optimized using Optuna hyperparameter tuning with early stopping and pruning.

- Learning rate (`lr`) was searched on a log-uniform scale between `1e-5` and `1e-2`.  
- Dropout rate (`dropout_rate`) was tested between `0.1` and `0.5`.  
- Weight decay (`weight_decay`) was tuned on a log-uniform scale between `1e-5` and `1e-2`.  
- The model was trained using mini-batches with a fixed batch size.  
- Binary cross-entropy loss with logits was used during training.  
- Early stopping with patience of 7 epochs was implemented to prevent overfitting.  
- Optuna’s Median Pruner was used to stop unpromising trials early.  
- The primary evaluation metric was the validation F1-score.

**Long Short-Term Memory Network (LSTM):**  
The LSTM model was similarly optimized via Optuna with early stopping and pruning.

- Learning rate (`lr`), dropout rate (`dropout_rate`), and weight decay (`weight_decay`) were tuned in similar ranges as the FFN.  
- The number of hidden units and layers were fixed based on prior experiments.  
- Training used mini-batches and binary cross-entropy loss with logits.  
- Early stopping with a patience of 5 epochs was applied.  
- Optuna’s Median Pruner was used to prune non-promising trials.  
- Validation F1-score was used as the target metric for optimization.

#### Meta-Learner  
The meta-learner in the stacking ensemble is a **Logistic Regression** model configured with default parameters optimized for binary classification. Unlike the base learners, explicit hyperparameter tuning was minimal or not performed for the meta-learner due to:

- The meta-learner’s relatively simple architecture requiring fewer hyperparameters.  
- Use of the `'lbfgs'` solver with L2 regularization by default, which generally provides robust performance.  
- The focus on tuning base learners where most performance gains are realized.

Cross-validation during stacking training (3-fold stratified K-Fold) implicitly ensures the meta-learner generalizes well without overfitting. Future improvements may explore tuning regularization strength (`C`) or solver types if needed.

#### Optimisation Strategy

- **Grid Search / Random Search:** Hyperparameter combinations for classical models were explored using Grid Search or Randomized Search with cross-validation on the training set.  
- **Optuna Bayesian Optimization:** The FFN and LSTM deep learning models were tuned using Optuna’s Tree-structured Parzen Estimator (TPE) sampler for efficient hyperparameter exploration, combined with pruning to reduce computation time.  
- **Evaluation Metric:** The F1-score for the positive (attack) class was used as the primary metric to balance precision and recall.  
- **Early Stopping:** Models supporting it (e.g., XGBoost, FFN, LSTM) used early stopping to prevent overfitting.  
- **Resource Considerations:** To manage computational cost, the search space was carefully selected based on domain knowledge and prior experiments, with pruning mechanisms further reducing wasted resources.

### Multi-Class Classifier

For the selected multi-class classifier, which is a **Stacking Ensemble**, hyperparameter optimization targeted tuning of individual base learners as well as the meta-learner to maximize overall multi-class predictive performance.

#### Base Learners and Hyperparameters Tuned

**Logistic Regression:**  
The Logistic Regression base model was tuned using `GridSearchCV` with a pipeline.

- Regularization strength inverse, tested across `[0.01, 0.1, 1, 10, 100]` (`C`)  
- Solver algorithms supporting multi-class classification tested: `'lbfgs'` and `'saga'` (`solver`)  
- Penalty set to `'l2'` to ensure robust multi-class performance (`penalty`)  
- Multi-class strategy set to `'multinomial'` (default with `'lbfgs'`)  

**Random Forest (RF):**  
The Random Forest base model was tuned using `GridSearchCV` with a pipeline.

- Number of trees, tested at 100, 200, and 300 (`n_estimators`)  
- Maximum depth of trees, tested at 10, 20, and unlimited (`max_depth`)  
- Minimum samples required to split a node, tested at 2 and 5 (`min_samples_split`)  
- Minimum samples required at a leaf node, tested at 1 and 2 (`min_samples_leaf`)  
- Number of features considered for the best split, tested with `'sqrt'` and `'log2'` (`max_features`)  

**XGBoost:**  
The XGBoost base model for multi-class classification was tuned using **Optuna** with a pipeline and stratified cross-validation.

- Number of boosting rounds (`n_estimators`) was searched between 100 and 2000 in steps of 100.  
- Maximum depth of trees (`max_depth`) was searched between 3 and 15.  
- Step size shrinkage (`learning_rate`) was searched on a log scale between 0.001 and 0.3 to control overfitting.  
- Subsample ratio of training instances (`subsample`) was searched between 0.5 and 1.0.  
- Subsample ratio of columns when constructing trees (`colsample_bytree`) was searched between 0.5 and 1.0.  
- Minimum loss reduction (`gamma`) was searched between 0.0 and 1.0 to control complexity.  
- L1 (`reg_alpha`) and L2 (`reg_lambda`) regularization parameters were searched on log scales from 1e-8 to 10.0 for weight penalties.  
- Multi-class objective was set to `'multi:softprob'` with the appropriate number of classes.  
- Early pruning of trials was enabled via Optuna's Median Pruner to improve efficiency.  
- Model performance was evaluated using stratified 3-fold cross-validation with the F1-macro score as the optimization metric.

#### Deep Learning Models

**Feedforward Neural Network (FFN):**  
The FFN model was optimized using a combination of manual tuning and automated hyperparameter search.

- Number of hidden layers, tested between 2 and 4 layers  
- Number of neurons per layer, ranging from 32 to 256  
- Activation functions tested included ReLU and LeakyReLU  
- Dropout rates tested at 0.1, 0.3, and 0.5 to prevent overfitting  
- Batch size experimented with values 32, 64, and 128  
- Learning rates tuned between 0.0001 and 0.01 using optimizers such as Adam and RMSprop  
- Early stopping implemented to halt training when validation loss plateaued  

**Recurrent Neural Network (RNN) / LSTM:**  
For sequential or time-dependent features, RNNs with LSTM units were utilized and tuned.

- Number of LSTM layers varied between 1 and 3  
- Hidden units per LSTM layer tested from 50 to 200  
- Dropout and recurrent dropout rates tested at 0.2 and 0.5  
- Sequence length input adjusted based on feature representation  
- Optimizers tested included Adam and RMSprop with learning rates between 0.0001 and 0.01  
- Early stopping used to prevent overfitting  

#### Meta-Learner

The meta-learner in the stacking ensemble is a **Logistic Regression** model configured for multi-class classification with default parameters. Specific considerations include:

- Multi-class strategy set to `'multinomial'` to handle multiple classes jointly.  
- Solver configured to `'lbfgs'` for efficient multi-class optimization.  
- Minimal explicit hyperparameter tuning was performed given the meta-learner’s relatively simple architecture and reliance on base learner outputs.

Cross-validation during stacking training (3-fold stratified K-Fold) ensured generalization of the meta-learner without overfitting. Future improvements could include tuning regularization strength (`C`) or exploring alternative meta-learners better suited for multi-class problems.

#### Optimisation Strategy

- **Grid Search / Random Search:** Hyperparameter combinations were explored via Grid Search or Randomized Search with cross-validation on the training data.  
- **Evaluation Metric:** The macro-averaged F1-score was used as the primary metric to balance precision and recall across all classes equally.  
- **Early Stopping:** Applied where supported (e.g., XGBoost, deep learning models) to prevent overfitting during training.  
- **Resource Considerations:** The search space was narrowed based on domain knowledge and prior experience to balance thoroughness and computational efficiency.

---
## 💡 Key Findings
### Key Results and Findings for the Binary Classifier
#### ROC Curve Comparison:
<p align="center">
  <a href="Binary ROC Curves">
    <img src="https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/images/roc_curve_comparison.png" alt="ROC Curves Comparison" width="800" />
  </a>
</p>

#### Precision Recall Curve Comparison: 
<p align="center">
  <a href="Binary PrecRecall Curves">
    <img src="https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/images/precision_recall_curve_comparison.png" alt="Prec-Recall Curves Comparison" width="800" />
  </a>
</p>

#### Top 3 Models (Binary Class):

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
#### ROC Curve Comparison:
<p align="center">
  <a href="Multi ROC Curves">
    <img src="https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/images/multi_micro_roc_curve_comparison.png" alt="ROC Curves Comparison" width="800" />
  </a>
</p>

#### Precision Recall Curve Comparison: 
<p align="center">
  <a href="Multi PrecRecall Curves">
    <img src="https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/images/multi_micro_pr_curve_comparison.png" alt="Prec-Recall Curves Comparison" width="800" />
  </a>
</p>

#### Top 3 Models (Multi-Class):

<div align="center">

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>F1-score<br>(Weighted)</th>
      <th>ROC AUC<br>(Micro-avg)</th>
      <th>PR AP<br>(Micro-avg)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random Forest (Untuned)</td>
      <td>0.9875</td>
      <td>0.9981</td>
      <td>0.9943</td>
    </tr>
    <tr>
      <td>Stacking Classifier</td>
      <td>0.9869</td>
      <td>0.9994</td>
      <td>0.9983</td>
    </tr>
    <tr>
      <td>Random Forest (Tuned)</td>
      <td>0.9864</td>
      <td>0.9994</td>
      <td>0.9983</td>
    </tr>
  </tbody>
</table>

</div>

The top-performing models — **Stacking Classifier**, **Random Forest (Untuned and Tuned)** — demonstrate strong and consistent performance across all key evaluation metrics:

- **Weighted F1-score**: In the range of **0.9864–0.9875**, indicating a well-balanced precision and recall across all classes, including minority ones.
- **Micro-average ROC AUC**: Approximately **0.9981–0.9994**, showing near-perfect discrimination capability across all classes.
- **Micro-average Precision-Recall Average Precision (AP)**: Between **0.9943–0.9983**, reflecting excellent predictive performance even in the presence of class imbalance.

#### Per-Class Performance Highlights:

The top models not only perform well overall, but also excel across individual attack classes:

- **BENIGN** and **DoS Hulk** consistently show near-perfect ROC AUC and Average Precision across tree-based and tuned deep learning models.
- **PortScan** and **DDoS** are also predicted with high confidence (ROC AUC ~0.998+, AP ~0.995) by most top models.
- **rare_attack** class — the most challenging due to its sparsity — still achieves high ROC AUC (~0.99+) and AP (~0.89–0.92) with **Stacking**, **RF (Tuned)**, and **XGBoost (Tuned)**, showing strong minority class sensitivity.

#### Model Family Observations:

**Tree-based models (Random Forest and XGBoost)** consistently outperform deep learning and linear models, likely due to:

- Robustness to noisy or redundant features in network traffic.
- Efficient handling of class imbalance and skewed distributions.
- Better generalization on structured tabular data, even without extensive tuning.

**Deep Learning models** show reasonable but comparatively lower performance. Despite effective hyperparameter tuning, further gains may require:

- Larger or augmented datasets for better generalization.
- More expressive architectures (e.g., CNNs or attention-based models).
- Feature engineering or dimensionality reduction to reduce noise.

**Logistic Regression** lags behind, with weaker performance on minority classes like **DDoS** and **rare_attack**, due to:

- Inability to model complex, non-linear relationships.
- Sensitivity to class imbalance, especially without advanced weighting strategies.

#### Final Recommendation:

The **Stacking Ensemble** model offers the best combination of **robustness**, **per-class accuracy**, and **overall predictive strength**, making it the most suitable candidate for deployment in **multi-class intrusion detection systems** using the CICIDS2017 dataset.

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
Due to compute and time constraints, the current pipelines have been developed and evaluated on a small stratified subset of the full CICIDS2017 dataset. Future work should focus on scaling the training and tuning processes to the entire dataset to potentially improve model robustness and generalization. Additionally, deeper hyperparameter tuning, especially for deep learning models, is necessary to unlock their full potential. Exploring more sophisticated deep learning architectures such as CNNs or advanced ensemble methods could further enhance performance for both binary and multi-class classification tasks.
### Data & Feature Refinement  
Given the subset limitation, expanding the dataset size could allow for development of new, domain-specific features and application of more advanced feature selection techniques. Investigating cost-sensitive learning approaches or more refined resampling strategies will help better address class imbalance challenges inherent in the full dataset. This could lead to improved model efficiency and interpretability.
### Productionization  
Building a complete end-to-end deployment pipeline remains a priority, including real-time inference, comprehensive model monitoring, and automated retraining workflows. These steps will be crucial to transition from the current proof-of-concept stage—limited by dataset size and tuning scope—to a scalable, practical operational solution suitable for real-world cybersecurity environments.

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

