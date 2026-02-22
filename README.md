# Credit Card Fraud Detection using Sampling Techniques

## Objective
The objective of this project is to analyze the impact of different sampling techniques on the performance of multiple machine learning models when applied to a highly imbalanced credit card fraud dataset. The study focuses on understanding how various data sampling strategies influence model accuracy, stability, and generalization.

---

## Dataset
The dataset used in this project consists of credit card transaction records labeled as:

- `0` → Legitimate Transaction  
- `1` → Fraudulent Transaction  

Due to the highly imbalanced nature of the dataset, class balancing and sampling techniques are applied before training the machine learning models.

**Dataset File:**

[Creditcard_data.csv](./Creditcard_data.csv)

## Sampling Techniques Used
The following five statistical sampling techniques were implemented:

1. **Simple Random Sampling**  
2. **Systematic Sampling**  
3. **Stratified Sampling**  
4. **Cluster Sampling**  
5. **Bootstrap Sampling**


## Model Evaluation Method
To assess model stability and generalization performance, **5-Fold Cross-Validation** was applied to all machine learning models.

## Machine Learning Models Used
The following five machine learning models were evaluated:

- Logistic Regression (M1)  
- Decision Tree (M2)  
- Random Forest (M3)  
- K-Nearest Neighbors (M4)  
- Naive Bayes (M5)  

## Project Structure
Sampling_Assignment/
│
├── main.py
├── Creditcard_data.csv
├── accuracy_results.csv
└── README.md

## License

© 2026 Arshdeep Kaur

This reopsitory is licensed under MIT License. See LICENSE for details.
