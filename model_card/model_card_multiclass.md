# Model Card: Stacking Classifier for Multi-Class Classification (CICIDS2017)

## Model Details

- **Model Name**: Stacking Classifier (Multi-Class)
- **Version**: 1.0  
- **Date**: 05-08-2025  
- **Model Type**: Ensemble Stacking Classifier  
- **Framework**: scikit-learn  
- **Owner/Creator**: Akhilesh Pokhariyal  
- **License**: Apache 2.0  
- **Contact**: akhilesh.pokhariyal@gmail.com  

---

## Model Description

**Input:**  
A vector of 78 numerical features derived from preprocessed network traffic, normalized and optionally transformed using PCA.

**Output:**  
Multi-class label for one of the following:
- `BENIGN`
- `DDoS`
- `DoS Hulk`
- `PortScan`
- `rare_attack` (aggregated rare attacks)

Also returns class probabilities for each prediction.

**Model Architecture:**  
An ensemble **Stacking Classifier** combining predictions from:
- Logistic Regression
- Random Forest (tuned)
- XGBoost (tuned)

The meta-learner is a Logistic Regression classifier.

---

## Intended Use

- **Use Case**: Detect and classify network traffic behavior (normal vs. different attack types)
- **Target Users**: Cybersecurity analysts, IT admins, IDS developers  
- **Not Suitable For**: Real-time applications or environments not resembling CICIDS2017

---

## Training Data

- **Dataset**: Subset of CICIDS2017 (original size: 2,827,876 rows × 79 columns)
- **Subset used**: 19,992 rows × 78 features (due to compute and time constraints)
- **Test set size**: 3,999 samples (randomly held out from the subset)
- **Features**: 78 preprocessed numerical features  
- **Class labels**: BENIGN, DDoS, DoS Hulk, PortScan, rare_attack
- **Preprocessing**: Standardization, PCA for dimensionality reduction, and SMOTE for class imbalance

> ⚠️ **Note**: The full CICIDS2017 dataset was not used for training or hyperparameter tuning. Model performance reflects evaluation on this smaller subset.

---

## Assumptions and Constraints

- Only a small, stratified subset of the CICIDS2017 dataset (19,992 samples) was used for the entire modeling pipeline — including preprocessing, training, and tuning — due to limited computational resources and time.
- Consequently, the reported performance metrics may not generalize to the full dataset or real-world deployment scenarios.
- The current model serves as a **proof-of-concept** and is **not optimized for production**. Further scaling, retraining on the full dataset, and extensive validation are required before operational use.

---

## Performance

**Validation**: Hold-out test set  
**Test Dataset**: Representative subset from CICIDS2017

### Metrics:
- **Accuracy**: 0.98  
- **Macro F1-score**: 0.93  
- **Weighted F1-score**: 0.98  
- **ROC AUC (micro)**: 0.9955  
- **PR AUC (micro)**: 0.9850  
- **Composite Rank**: 3.5

### Classification Report:

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| BENIGN       | 1.00      | 0.98   | 0.99     | 3213    |
| DDoS         | 0.98      | 0.98   | 0.98     | 181     |
| DoS Hulk     | 0.96      | 0.98   | 0.97     | 325     |
| PortScan     | 0.99      | 1.00   | 0.99     | 225     |
| rare_attack  | 0.58      | 0.85   | 0.69     | 55      |

### Confusion Matrix:

| Actual \ Predicted | BENIGN | DDoS | DoS Hulk | PortScan | rare_attack |
|--------------------|--------|------|----------|----------|-------------|
| **BENIGN**         | 3160   | 3    | 13       | 3        | 34          |
| **DDoS**           | 3      | 178  | 0        | 0        | 0           |
| **DoS Hulk**       | 5      | 0    | 320      | 0        | 0           |
| **PortScan**       | 0      | 0    | 0        | 225      | 0           |
| **rare_attack**    | 6      | 0    | 2        | 0        | 47          |


---

## Limitations

- Trained on CICIDS2017 only — may not generalize to other network types.
- Model trained on a small subset (~0.7%) of the original CICIDS2017 dataset, which may impact generalizability to the full spectrum of attack types and network traffic behaviors.
- Performance on rare_attack class is limited by data sparsity.
- Additional work needed for real-time inference.
- Relies on specific preprocessing (PCA, SMOTE).

---

## Trade-offs

- **Accuracy vs Interpretability**: Ensemble models reduce transparency.
- **Recall vs Precision**: Prioritizes recall on attack types to reduce false negatives.
- **Compute Cost**: Higher due to model complexity and ensemble strategy.

---

## Ethical Considerations

- May underperform on unseen attack types or network configurations.
- False positives may lead to unnecessary alerts.
- Human review recommended before acting on predictions.

---

## Recommendations

- ✅ Retrain periodically with up-to-date network traffic and threats.  
- ✅ Use with anomaly detection or rule-based systems for robust protection.  
- ✅ Validate model against production traffic before deployment.  
- ❌ Avoid using it as a black-box decision tool without oversight.  

---

## Checkpoints and Artifacts

- **Final model**: `./model_checkpoints/final_best_multi_class_model_stacking_classifier.pkl`  
- **Best model name**: `./model_checkpoints/best_model_name_multi_class.pkl`  
- **Predictions**: `./model_checkpoints/final_y_pred_multi_class.npy`  
- **Probabilities**: `./model_checkpoints/final_y_prob_multi_class.npy`  

---

## References

- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)  
- [Model Card Toolkit (Google)](https://github.com/tensorflow/model-card-toolkit)

