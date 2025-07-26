# 🗣️ Threat Detection in Network Traffic

## 📌 Project Overview


---

## 📖 Notebooks
Explore our Jupyter notebooks with in-depth analysis and insights:

- **Model and insights** – Data analysis, feature engineering, and model building.
- **Ideas for implementation** – How this research can be applied in real-world settings.
- **En español: modelo y conclusiones** – Spanish version with key findings and model details.

---
  

---

## 🚀 Project Workflow
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

- ✅ Increasing accessibility and funding for Dutch language courses – Since formal enrollment is the strongest predictor of proficiency.  
- ✅ Offering tailored curricula for high-risk language groups – Non-Indo-European speakers (e.g., Arabic, Chinese, Turkish) face significant challenges.  
- ✅ Promoting early exposure programs – Younger arrivals have better outcomes; early intervention can boost long-term success.  
- ✅ Encouraging multilingual education – Bilingual individuals perform better, suggesting benefits to fostering multilingual skills.  
- ✅ Focusing on active language learning over general schooling – Dutch language training is far more effective than formal education alone.  

---

## 🔬 Future Work
- Expand data collection to include more diverse linguistic backgrounds.  
- Investigate motivational and socioeconomic factors influencing proficiency.  
- Conduct a longitudinal study to track language acquisition before and after course enrollment.  
- Enhance predictive models using AI for personalized language learning plans.  

---

## 🛠️ Tech Stack
- **Python & Libraries:**  
Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, joblib, Optuna, PyTorch
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



