# Model Card: Stacking Classifier for Binary Classification (CICIDS2017)

## Model Details

- **Model Name**: Stacking Classifier
- **Version**: 1.0
- **Date**: 04-08-2025
- **Model Type**: Ensemble Stacking Classifier
- **Framework**: scikit-learn
- **Owner/Creator**: Akhilesh Pokhariyal
- **License**: Apache 2.0
- **Contact**: akhilesh.pokhariyal@gmail.com

---

## Model Description

**Input:**  
The model takes as input a feature vector of length 78, representing preprocessed network traffic data, including derived statistical and protocol-specific features. The features are normalized and optionally transformed using PCA before classification.

**Output:**  
The output is a binary classification label indicating whether the input network traffic is **BENIGN** or **ATTACK**. Additionally, the model outputs the probability score for the positive class (ATTACK) to enable threshold tuning or further analysis.

**Model Architecture:**  
The model is an ensemble Stacking Classifier combining multiple base learners (e.g., Random Forest, XGBoost, Logistic Regression) whose predictions are combined using a meta-classifier for improved predictive performance and robustness.

---

## Intended Use

- **Primary use case**: Network intrusion detection — binary classification of network traffic as BENIGN or ATTACK.
- **Users**: Security analysts, cybersecurity systems, researchers.
- **Not intended for**: Real-time deployment without optimization; environments with drastically different network traffic than CICIDS2017.

---

## Training Data

- **Dataset**: CICIDS2017
- **Training set size**: 226,230 samples, 78 features
- **Test set size**: 56,558 samples, 78 features
- **Class names**: BENIGN, ATTACK
- **Preprocessing**: PCA for dimensionality reduction, SMOTE for class imbalance, standard normalization.
- **Class Balance (Test set)**: Approx. 80% BENIGN, 20% ATTACK
- **Data Source**: [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## Performance

- **Metrics**:
  - F1 (Positive class - ATTACK): 0.9864
  - Precision (ATTACK): 0.98
  - Recall (ATTACK): 0.99
  - Accuracy: 0.99
- **Classification Report:**

  | Class  | Precision | Recall | F1-Score | Support |
  |--------|-----------|--------|----------|---------|
  | BENIGN | 1.00      | 1.00   | 1.00     | 45,427  |
  | ATTACK | 0.98      | 0.99   | 0.99     | 11,131  |

- **Confusion Matrix:**

  |                | Predicted BENIGN | Predicted ATTACK |
  |----------------|------------------|------------------|
  | Actual BENIGN  | 45,230           | 197              |
  | Actual ATTACK  | 108              | 11,023           |

- **Validation Method**: Hold-out test set from CICIDS2017.
- **Test Dataset**: Same as above, representative of real-world traffic in CICIDS2017.

---

## Limitations

- The model was trained and evaluated exclusively on CICIDS2017 data and may not generalize well to other network environments or attack types.
- May produce false positives or negatives on rare or novel attack patterns.
- Not optimized for real-time inference — additional engineering required.
- Assumes input features are preprocessed identically as in training (PCA, SMOTE).

---

## Trade-offs

- **Accuracy vs. Interpretability**: High performance achieved via stacking ensemble reduces model interpretability.
- **Precision vs. Recall**: Slight preference toward high recall on ATTACK class to minimize missed detections at some cost of false positives.
- **Resource Usage**: Ensemble stacking requires more compute and memory than simpler models, which may limit deployment on resource-constrained devices.

---

## Ethical Considerations

- Potential bias if training data does not represent all attack types or network scenarios.
- Steps taken to mitigate imbalance with SMOTE, but ongoing monitoring and retraining recommended.
- Model decisions should be complemented with human analyst review where possible.

---

## Caveats and Recommendations

- Deploy only in environments similar to CICIDS2017 network traffic distribution.
- Retrain regularly to account for new attack vectors or network changes.
- Use with additional anomaly detection layers or alerting mechanisms for robust defense.

---

## Checkpoints and Artifacts

- Best model checkpoint saved at: `./model_checkpoints/best_binary_model_name.pkl`
- Final best model saved at: `./model_checkpoints/final_best_binary_model_stacking_classifier.pkl`
- Final predictions saved at: `./model_checkpoints/final_binary_y_pred.npy`
- Final positive class probabilities saved at: `./model_checkpoints/final_binary_y_prob_pos_class.npy`

---

## References

- [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Model Card Toolkit (Google)](https://github.com/tensorflow/model-card-toolkit)








See the [example Google model cards](https://modelcards.withgoogle.com/model-reports) for inspiration. 

## Model Description

**Input:** Describe the inputs of your model 

**Output:** Describe the output(s) of your model

**Model Architecture:** Describe the model architecture you’ve used

## Performance

Give a summary graph or metrics of how the model performs. Remember to include how you are measuring the performance and what data you analysed it on. 

## Limitations

Outline the limitations of your model.

## Trade-offs

Outline any trade-offs of your model, such as any circumstances where the model exhibits performance issues. 
