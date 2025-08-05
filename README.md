# Employee-Attrition-Analysis-and-Prediction


#  Employee Attrition Analysis and Prediction

This project helps predict whether an employee will **stay** or **leave** a company using machine learning models. It uses a dataset with employee information and builds a prediction system with an interactive Streamlit app.

---

##  What This Project Does

- Predicts if an employee will **Stay** or **Leave**
- Uses machine learning models like **Logistic Regression**
- Adds new features like:
  - `EngagementScore` (based on satisfaction and involvement)
  - `TenureCategory` (based on years at the company)
- Visualizes trends using:
  - Bar Charts
  - Line Charts
  - Scatter 
- Provides a **Streamlit dashboard** for predictions

---

## 🛠 Tools & Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- XGBoost (optional)
- Pickle (for saving models)
- LabelEncoder, StandardScaler

---

## 📁 Project Folder Structure

Employee-Attrition/
│
├── data/
│ └── Employee-Attrition.csv
│
├── model/
│ ├── logistic_model.pkl
│ ├── scaler.pkl
│ └── encoders/
│ ├── gender_encoder.pkl
│ ├── department_encoder.pkl
│ └── ...
│
├── app.py ← Streamlit web app
├── train_model.py ← Script to train & save the model
├── requirements.txt ← Python packages
└── README.md ← This file
