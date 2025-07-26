# CICIDS2017 Dataset Datasheet

## Overview

The CICIDS2017 dataset is a publicly available benchmark dataset created by the Canadian Institute for Cybersecurity. It contains realistic network traffic data capturing benign and malicious activities for intrusion detection system research. The dataset includes labeled records representing multiple attack types and normal traffic.

## Motivation

The dataset was designed to provide a comprehensive and realistic network intrusion detection dataset with diverse attack scenarios and up-to-date traffic patterns. It aims to support research in network security, anomaly detection, and machine learning-based intrusion detection.

## Composition

- Contains network traffic data with approximately 3.1 million records.
- Each record is labeled as benign or one of several attack types:
  - Benign
  - Brute Force
  - Botnet
  - DDoS
  - DoS
  - Web Attacks
  - Infiltration
  - Port Scan
  - Heartbleed
- Data size: Around 10 GB total (depending on subset).
- Features include flow-based statistics such as duration, packet counts, byte counts, and more.

## Distribution

- Benign traffic: ~2,830,000 records (91%)
- DDoS: ~280,000 records (9%)
- Other attacks: Smaller proportions, from a few hundred to a few thousand records per type.
- Dataset is distributed in CSV format with labeled flows.
- Dataset is publicly available via the Canadian Institute for Cybersecurity website and mirrors such as Kaggle.

## Collection Process

- Data was captured using a real testbed network environment simulating both normal user behavior and malicious activities.
- Network traffic was generated over several days in July 2017.
- Attacks were launched in a controlled manner to mimic real-world scenarios.
- Tools such as CICFlowMeter were used to extract flow-based features from raw network packets.

## Preprocessing and Cleaning

- Raw network packet captures (pcap files) were processed to generate flow statistics.
- Anonymization was applied to IP addresses to protect privacy.
- Incomplete or corrupted flows were removed.
- Labels were assigned based on timestamps and attack scenarios.

## Uses

- Primarily used for training and evaluating intrusion detection systems (IDS).
- Useful for anomaly detection, classification, and cybersecurity research.
- Can support benchmarking of machine learning models on network security tasks.
- Suitable for educational purposes and tutorials on network traffic analysis.

## Maintenance

- Dataset was created and maintained by the Canadian Institute for Cybersecurity.
- No official updates have been released since initial publication in 2017.
- Community contributions and derived datasets exist but are maintained independently.

## Ethical Considerations

- All personally identifiable information (PII) was anonymized before dataset release.
- The dataset simulates malicious activity in a controlled environment; no real user data was compromised.
- Users should consider ethical implications when applying the dataset to real-world scenarios.
- Proper attribution to dataset creators is required.

## Licensing and Access

- The dataset is freely available for academic and research use.
- License terms can be found on the CIC website: https://www.unb.ca/cic/datasets/ids-2017.html
- Redistribution is permitted with attribution; commercial use may require permission.

## Limitations

- Dataset reflects network conditions and attack patterns from 2017; may not fully represent modern traffic.
- Imbalanced class distribution with majority benign traffic.
- Attack types do not cover all possible real-world threats.
- Some flows may not capture full attack payloads.
- Results using this dataset may not generalize perfectly to production networks.

---

*For more information, visit the Canadian Institute for Cybersecurity’s official CICIDS2017 page: https://www.unb.ca/cic/datasets/ids-2017.html*
