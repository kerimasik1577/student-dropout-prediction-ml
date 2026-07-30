# Student Dropout Prediction & Early Warning System


An end-to-end Machine Learning and Deep Learning pipeline designed to identify higher education students at risk of dropping out. By reforming the target structure, engineering non-linear interaction features, and optimizing decision thresholds, this system achieves a 91.52% Accuracy and an 87.62% Dropout Recall rate.

Project Overview & Key Highlights

Target Refactoring (Binary Reformulation): Removed ambiguous boundary noise caused by the Enrolled class, refactoring the problem into a clear binary decision (Dropout vs. Graduate).

Feature Engineering: Extracted top-5 critical features, generating non-linear interaction terms (products and ratios) alongside row-wise statistical aggregations (mean, std, median) to expand model capacity to 24 features.

Cost-Sensitive Threshold Tuning: Applied a custom decision threshold (0.65 Graduate Cutoff) to minimize False Negatives (unidentified dropouts), boosting Dropout Recall from ~83% to 87.62%.

Cross-Validation Reliability: Validated stability across all models using 10-Fold Stratified Cross-Validation (91.52% ± 1.01%).

Model Performance & Benchmark (10-Fold Stratified CV)

Quickstart & Inference Pipeline

You can make real-time risk predictions using the saved model artifacts:

import pandas as pd
from src.inference import predict_student_risk

# Load new raw student data
new_students = pd.read_csv('data/data.csv').head(5)

# Predict risk status using custom threshold (0.65)
risk_report = predict_student_risk(new_students, threshold=0.65)
print(risk_report)


🛠️ Tech Stack & Tools

Language: Python 3.10+

Machine Learning & DL: LightGBM, Scikit-Learn, PyTorch

Model Explainability: SHAP (TreeExplainer)

Data Manipulation & Viz: Pandas, NumPy, Matplotlib, Seaborn

Persistence: Joblib
