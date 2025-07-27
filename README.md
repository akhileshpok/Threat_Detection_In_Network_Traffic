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
Explore my Jupyter notebooks with in-depth analysis, plots and insights:

- [Data Prep Script](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/notebooks/Capstone_project_data_prep.ipynb) - Notebook for data preparation.
- [BinaryClass ML pipeline](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/notebooks/CICIDS2017_BinaryClass_Pipeline.ipynb) - Notebook for the Binary Class ML pipeline. 
- [MultiClass ML pipeline](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/notebooks/CICIDS2017_MultiClass_Pipeline.ipynb) - Notebook forthe Multi-Class ML pipeline.

---
  

## ⚙️ Project Workflow
The project follows a structured machine learning pipeline:

1. Data Loading – Load and preprocess the dataset.  
2. Exploratory Data Analysis – Identify trends, correlations, and distributions.  
3. Train/Test Split – Partition the dataset for model evaluation.  
4. Feature Engineering – Transform categorical and numerical features.  
5. Regression Modeling – Train multiple regression models.  
6. Hyperparameter Tuning – Optimize models for better performance.  
7. Model Selection – Choose the best-performing model.  
8. Model Interpretation – Use SHAP values to explain feature importance.  
9. Results & Analysis – Evaluate model accuracy and key findings.  
10. Model Deployment – Save and export the trained model.  

---

## 📊 Dataset Datasheet
[Datasheet](https://github.com/akhileshpok/Threat_Detection_In_Network_Traffic/blob/main/data_sheet/CICIDS2017_Dataset_Datasheet.md)

## 📊 Model Card

## 📊 Key Findings
- Enrollment in Dutch language courses is the strongest predictor of speaking proficiency. Structured learning is significantly more effective than passive exposure.  
- Native speakers of Germanic languages (e.g., German, English, Swedish) perform best, while non-Germanic Indo-European and non-Indo-European speakers face greater challenges.  
- Monolingual individuals struggle more, whereas those with a second language, especially a Germanic one, have an advantage.  
- Age of arrival matters – Younger arrivals tend to perform slightly better in Dutch proficiency tests.  
- Length of residence alone has a weak effect – Long-term residence only improves proficiency when combined with active learning.  
- General formal education has minimal impact – Learning Dutch through structured courses is more influential than overall schooling background.  

---

## 🎯 Business Recommendations
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



