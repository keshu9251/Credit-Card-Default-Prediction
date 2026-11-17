# Credit Card Default Prediction Using Machine Learning

## Project Overview
This project uses machine learning to predict whether a credit card customer is likely to miss their next month's payment. We trained and compared three machine learning algorithms to find the best-performing model.

## Student Details
**Student 1:** Keshav Agarwal  
**Roll Number:** 253301024

**Student 2:** Anirudh Kalla

**College:** Lachoo Memorial College  
**Course:** MCA – 3rd Semester

## Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset:** Default of Credit Card Clients
- **Original Records:** 30,000
- **Records After Removing Duplicates:** 29,965
- **Input Features:** 23
- **Target:** DEFAULT (0 = No Default, 1 = Default)

**Dataset Link:**  
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

## Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree
3. Random Forest

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 68.65% | 37.56% | 62.97% | 47.06% | 71.62% |
| Decision Tree | 77.59% | 49.43% | 55.13% | 52.12% | 75.20% |
| Random Forest | 79.18% | 53.00% | 51.88% | 52.44% | 76.59% |

**Final Model:** Tuned Random Forest pipeline, selected after 5-fold cross-validation and GridSearchCV. See tuned test results below.



## Live Application
[Open the CreditGuard Streamlit application](https://credit-card-default-prediction-keshav.streamlit.app/)

## Cross-Validation and Hyperparameter Tuning
- Method: Stratified 5-fold cross-validation
- Tuning: GridSearchCV
- Parameter combinations: 16 (80 total model fits)
- Best cross-validation ROC-AUC: 0.7801
- Best parameters: n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2

## Tuned Model Test Results
- Accuracy: 78.09%
- Precision: 50.45%
- Recall: 55.51%
- F1-score: 52.85%
- ROC-AUC: 0.7679

The tuned Random Forest pipeline is saved as `credit_card_default_model.pkl` and loaded by the Streamlit application. The initial Random Forest achieved 79.18% accuracy; the tuned model has slightly lower accuracy but improved recall and ROC-AUC.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Jupyter Notebook

## Project Files
- `app.py` – Streamlit application
- `Credit_Card_Default_Prediction.ipynb` – Model training and analysis
- `credit_card_default_model.pkl` – Saved machine learning model
- `requirements.txt` – Required Python libraries

## How to Run

1. Download or clone the project.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Limitations
This is an academic project. Predictions are estimates and should not be used as the only basis for real financial decisions.

## Project Team
- Keshav Agarwal
- Anirudh Kalla

MCA – 3rd Semester  
Lachoo Memorial College
