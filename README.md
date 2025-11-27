**📊 Credit Risk Modeling (Multi-Class Classification)**

This project focuses on building a Credit Risk Prediction System that classifies applicants into one of four priority categories — P1, P2, P3, P4 — where P1 represents the highest creditworthiness. The goal is to help financial institutions make informed decisions for credit card or loan approvals using machine learning.


**📁 Project Structure**
Credit Risk Modeling/
│── Datasets/                   # Raw and processed dataset files
│── Notes/                      # Notes, EDA thoughts, references, formulas
│── Credit_Risk_Modeling.py     # Main model training and evaluation script
│── README.md                   # Project documentation
│── .gitignore                  # Git ignored files list


**🚀 Project Overview**
This project builds a machine learning pipeline to:
 -> Load & preprocess credit applicant data
 -> Handle missing values, encode categorical features, and scale numerical values
 -> Train multiple ML models
 -> Compare their performance
 -> Predict the credit risk class (P1 to P4)
 -> This helps banks prioritize applicants based on predicted repayment capability.


 **🧠 Machine Learning Models Used**
    🔹 Random Forest Classifier
    🔹 XGBoost Classifier
    🔹 Decision Tree Classifier


**⚙️ Preprocessing Steps**
  -> Cleaning missing values
  -> Handling categorical features
  -> Outlier detection & capping
  -> Feature scaling (if needed)
  -> Train-test split
  -> Addressing class imbalance (if present)

  
**📈 Evaluation Metrics**
    -> Since this is a multi-class classification problem (P1–P4), results were evaluated using:
        -> Accuracy
        -> Classification Report (Precision, Recall, F1-score)
        -> Confusion Matrix
        -> Cross-validation



**📚 Folder Details**
    📁 Datasets/
        -> Contains all dataset files (raw and processed).

    📁 Notes/
        -> Contains your notes, observations, formulas, and research.

    📄 Credit_Risk_Modeling.py
        Main Python script including:
            -> Preprocessing
            -> Model training
            -> Evaluation
            -> Final predictions


**🏁 Conclusion**
This project demonstrates how machine learning can be used for credit risk classification, helping financial institutions automate credit assessment and make data-driven approval decisions.