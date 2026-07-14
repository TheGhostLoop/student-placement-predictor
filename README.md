# 🎓 Student Placement Predictor

## 📌 Overview

A Machine Learning web application that predicts whether a student is likely to be placed based on academic performance, technical skills, internships, and other attributes.

The application compares multiple machine learning models and uses the best-performing model (Gradient Boosting Classifier) for prediction. It is deployed using Streamlit and provides prediction probabilities along with model evaluation visualizations.

## 🚀 Features

- Predicts student placement status
- Displays prediction probability
- Compares multiple ML algorithms
- Feature Importance visualization
- Confusion Matrix visualization
- Interactive Streamlit interface

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib

## 📊 Machine Learning Workflow

1. Data Preprocessing
2. One-Hot Encoding
3. ColumnTransformer
4. Train/Test Split
5. Model Training
6. Model Evaluation
7. Streamlit Deployment

## 📈 Model Comparison

| Model | Accuracy |
|---|---|
| Logistic Regression | 84.93% |
| Decision Tree | 99.95% |
| Random Forest | 99.95% |
| KNN | 89.66% |
| SVC | 84.04% |
| Gradient Boosting | **100.00%** |

## 📷 Application Preview

## Home page

## Prediction Result
## Accuracy Comparison
<img width="1460" height="720" alt="image" src="https://github.com/user-attachments/assets/3410910c-f672-4fd3-8b39-d9a65b0a6a8a" />

## Feature Importance
<img width="1460" height="863" alt="image" src="https://github.com/user-attachments/assets/bc4f38b6-75d4-4d6f-8349-fea8e9707b47" />

## Confusion Matrix
<img width="1460" height="1368" alt="image" src="https://github.com/user-attachments/assets/7662c9da-951b-48fe-831c-09913efbce08" />

## 📂 Project Structure

```
student-placement-predictor/
│
├── app.py
├── placement_model.pkl
├── requirements.txt
├── accuracy_comparison.png
├── feature_importances.png
├── confusion_matrix.png
├── README.md
└── dataset.csv
```

## ▶️ Installation

```bash
git clone https://github.com/yourusername/student-placement-predictor.git
cd student-placement-predictor
pip install -r requirements.txt
streamlit run app.py
```





