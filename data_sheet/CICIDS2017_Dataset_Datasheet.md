# CICIDS2017 Dataset Datasheet

## Dataset Description

The CICIDS2017 dataset is a publicly available network intrusion detection dataset created by the Canadian Institute for Cybersecurity. It contains anonymized network traffic data collected over several days, featuring both benign and malicious flows, simulating real-world network conditions.

The dataset was designed to provide a comprehensive benchmark for intrusion detection systems (IDS) and machine learning models aimed at detecting cyber attacks.

## Motivation

The dataset was created to facilitate research on network intrusion detection by providing realistic, labeled traffic data that includes a variety of attack types alongside normal traffic, allowing for the development and evaluation of effective detection techniques.

## Dataset Composition

- **Instances:** Approximately 2.8 million network flows.
- **Features:** Around 80 features per flow, including flow duration, protocol, source/destination IP and ports, packet counts, bytes, and various statistical features.
- **Labels:** Each flow is labeled as benign or belonging to one of several attack categories.

## Labels and Class Distribution

The dataset's primary target is to classify network flows as either benign or malicious.

- **Primary Target:** Binary classification (Benign vs. Attack).
- **Original Labels:** The dataset originally contains 15 distinct labels, including:
  - **Benign:** Normal network traffic.
  - **Attacks:** PortScan, DoS (GoldenEye, Hulk, Slowhttptest, Slowloris), DDoS, SSH-Patator, FTP-Patator, Web Attack (Brute Force, SQL Injection, XSS), Infiltration, Bot, and Heartbleed.

## Data Collection Process

The data was collected using network traffic capture tools over multiple days in a controlled environment with simulated attacks to reflect realistic network behavior. The attackers and benign users generated traffic simultaneously to mimic real network conditions.

## Preprocessing and Cleaning

- Anonymization of sensitive information (e.g., IP addresses).
- Removal of incomplete or corrupted flows.
- Labeling of flows based on attack type.
- Handling of missing and infinite values as needed for model training.

## Key Characteristics & Challenges

- **Severe Class Imbalance:** Approximately 80% of the data is benign traffic, while the remaining 20% represents various types of attacks. This imbalance necessitates evaluation metrics like F1-score, ROC AUC, and Average Precision over simple accuracy.
- **Redundancy and Duplication:** The dataset contains a high number of duplicate or redundant flow records, which can impact model training and validation.
- **Feature Complexity:** The dataset features a mix of numerical, categorical, and time-series-like data (e.g., flow duration, timestamp), challenging for various machine learning approaches.
- **Missing Values:** Some missing and infinite values exist, requiring careful preprocessing.

## Ethical Considerations

- **Privacy:** The dataset contains anonymized network traffic data. All personally identifiable information (PII) such as IP addresses has been anonymized to protect the privacy of individuals and organizations.
- **Consent:** The data was collected in controlled environments with simulated attacks, minimizing risks related to consent. However, users of the dataset should ensure compliance with local laws and regulations when applying or sharing derived models.
- **Usage Restrictions:** The dataset is intended solely for research and educational purposes in cybersecurity. It should not be used to facilitate malicious activities or unauthorized network intrusions.
- **Bias and Fairness:** The dataset reflects specific network environments and attack types which may not generalize to all real-world networks, potentially limiting the fairness and effectiveness of models trained solely on this data.
- **Responsible Use:** Researchers are encouraged to consider the implications of deploying models trained on this data, especially regarding false positives/negatives, and the potential impact on real network users.


## Distribution

The dataset is publicly available on the Canadian Institute for Cybersecurity website and on Kaggle. The data can be downloaded in CSV format for ease of use with machine learning frameworks.

## Usage

The dataset is commonly used for benchmarking intrusion detection algorithms, machine learning model training, and cybersecurity research.

## Maintenance

The CICIDS2017 dataset is maintained by the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick. It is a static dataset, and no versioned updates or maintenance cycles have been published since its release.

- **Maintainer**: Canadian Institute for Cybersecurity (CIC), UNB  
- **Update Frequency**: No regular updates; dataset is static  
- **Version Control**: Not applicable  
- **Issue Reporting**: Users may contact CIC directly; no public issue tracker  
- **Corrections**: No formal correction process; cleaning is user-managed  
- **Deprecation**: No known deprecation or replacement plans  

Future datasets may be released separately through CIC, but updates to CICIDS2017 itself are unlikely.

## Citation

If you use this dataset, please cite the original creators:

> Zeidanloo, H. R., & Manaf, A. A. (2017). CICIDS2017 Dataset: Canadian Institute for Cybersecurity Intrusion Detection Dataset.  
> Schepens, J., van Hout, R., & Jaeger, S. (2020). CICIDS2017: Intrusion Detection Data Set. Available on [Kaggle](https://www.kaggle.com/datasets/).

## License

Please check the dataset's official page for licensing terms. Typically, this dataset is available for research and educational use.

### Additional Notes

- Preprocessing is essential due to missing and infinite values.
- Handling class imbalance improves model reliability.
- Feature engineering may be required to optimize performance.

---

*This datasheet is intended to provide a thorough understanding of the CICIDS2017 dataset for researchers and practitioners in cybersecurity and machine learning.*

*For more information, visit the Canadian Institute for Cybersecurity’s official CICIDS2017 page: https://www.unb.ca/cic/datasets/ids-2017.html*
