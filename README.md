

\# Credit Card Default Prediction Using Machine Learning



\## Project Overview

This project predicts whether a credit card customer is likely to default on the next month's payment using machine learning classification algorithms.



\## Student Details

\- Name: Keshav Agarwal

\- College: Lachoo Memorial College

\- Course: MCA – 3rd Semester

\- Roll Number: 253301024



\## Dataset

\- Source: UCI Machine Learning Repository

\- Dataset: Default of Credit Card Clients

\- Records: 30,000 originally; 29,965 after duplicate removal

\- Input Features: 23

\- Target: DEFAULT (0 = No Default, 1 = Default)



Dataset Link:

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients



\## Machine Learning Algorithms

1\. Logistic Regression

2\. Decision Tree

3\. Random Forest



\## Model Performance



| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |

|---|---:|---:|---:|---:|---:|

| Logistic Regression | 68.65% | 37.56% | 62.97% | 47.06% | 71.62% |

| Decision Tree | 77.59% | 49.43% | 55.13% | 52.12% | 75.20% |

| Random Forest | 79.18% | 53.00% | 51.88% | 52.44% | 76.59% |



Random Forest was selected for the application based on its overall performance.



\## Technologies Used

\- Python

\- Pandas and NumPy

\- Matplotlib and Seaborn

\- Scikit-learn

\- Streamlit

\- Plotly

\- Joblib

\- Jupyter Notebook



\## Project Files

\- `app.py` – Streamlit application

\- `Credit\_Card\_Default\_Prediction.ipynb` – Model development and analysis

\- `credit\_card\_default\_model.pkl` – Saved Random Forest model

\- `requirements.txt` – Python dependencies



\## Run the Project Locally



1\. Clone or download this repository.

2\. Install dependencies:



&#x20;  `pip install -r requirements.txt`



3\. Run the application:



&#x20;  `streamlit run app.py`



\## Limitations

This project is an academic machine learning prototype. Its predictions are estimates and should not be used as the sole basis for real financial or lending decisions.



Cross-validation, systematic hyperparameter tuning and public deployment are pending completion.



\## Student

Keshav Agarwal  

MCA – 3rd Semester  

Lachoo Memorial College



