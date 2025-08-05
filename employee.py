#------------------------------------------------------EMPLOYEE ATTRITION PROJECT-----------------------------------------------------------------
#_________________________________________________________________________________________________________________________________________________

#  Import Libraries

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score,
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, f1_score
)
import pickle
import joblib
import os
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
#_________________________________________________________________________________________________________________________________________________
#  Load Dataset --------------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("C:/Users/Yokesh/guvi mini projects/Employee-Attrition-Analysis-and-Prediction/Employee-Attrition - Employee-Attrition.csv")

print(df)

df.head()

#_________________________________________________________________________________________________________________________________________________
# 3. check for missing values---------------------------------------------------------------------------------------------------------------------

print("missing values per column:\n", df.isnull().sum())

df.info()


#------------------------------------------------------------------------------------------------------------------------------------------------

#Exploratory Data Analysis

#Bar Plot — Count of Attrition
plt.figure(figsize=(6, 4))
sns.countplot(x='Attrition', data=df, palette='Set2')
plt.title('Employee Attrition Count')
plt.xlabel('Attrition (Yes = Left, No = Stayed)')
plt.ylabel('Number of Employees')
plt.tight_layout()
plt.savefig('eda_attrition_bar.png')


# Distribution Plot: Monthly Income
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='MonthlyIncome', kde=True, bins=30, color='skyblue')
plt.title('Distribution of Monthly Income')
plt.xlabel('Monthly Income')
plt.tight_layout()
plt.savefig('eda_income_distribution.png')


#________________________________________________________________________________________________________________________________________________
#  Feature Engineering---------------------------------------------------------------------------------------------------------------------------
df['EngagementScore'] = df[['JobInvolvement', 'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']].mean(axis=1)

def tenure_category(years):
    if years < 3:
        return 'Short'
    elif years < 6:
        return 'Medium'
    else:
        return 'Long'

df['TenureCategory'] = df['YearsAtCompany'].apply(tenure_category)
#------------------------------------------------------------------------------------------------------------------------------------------------
#  Drop unnecessary columns (e.g., 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours')
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df.drop(columns=drop_cols, inplace=True, errors='ignore')

#  Separate features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition'].apply(lambda x: 0 if x == 'Yes' else 1)
print(y)
print(df)
#------------------------------------------------------------------------------------------------------------------------------------------------

# 🔹 Encode categorical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

#------------------------------------------------------------------------------------------------------------------------------------------------

# 🔹 Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Save processed data (optional preview)
print(" Data preprocessing complete.")
print("Total features used:", X.shape[1])

print(df)

#------------------------------------------------------------------------------------------------------------------------------------------------


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


#------------------------------------------------------------------------------------------------------------------------------------------------
# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate
model_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    model_scores[name] = {
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC AUC": roc
    }

# Display results
results_df = pd.DataFrame(model_scores).T
print(" Model Evaluation Results:")
print(results_df)

# Select best model (based on F1 Score)
best_model_name = results_df["F1 Score"].idxmax()
best_model = models[best_model_name]
print(f"\n Best Model: {best_model_name}")

print("Target label distribution:\n", y.value_counts())


# Save the best model, encoders, and scaler

os.makedirs('models',exist_ok=True)

pickle.dump(best_model, open("models/best_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(encoders, open("models/encoders.pkl", "wb"))
#-----------------------------------------------------------------------------------------------------------------------------------------------
# streamlit UI

# Load trained model and preprocessing tools
model = pickle.load(open('C:/Users/Yokesh/guvi mini projects/Employee-Attrition-Analysis-and-Prediction/models/best_model.pkl', 'rb'))
scaler = pickle.load(open('C:/Users/Yokesh/guvi mini projects/Employee-Attrition-Analysis-and-Prediction/models/scaler.pkl', 'rb'))
encoders = pickle.load(open('C:/Users/Yokesh/guvi mini projects/Employee-Attrition-Analysis-and-Prediction/models/encoders.pkl', 'rb'))



st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("👩‍💼 Employee Attrition Prediction (Full Feature Dashboard)")
st.markdown("Enter the employee details below to predict attrition risk:")

# --- Collect all 32 feature inputs ---
with st.form("attrition_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.slider("Age", 18, 60, 30)
        BusinessTravel = st.selectbox("Business Travel", encoders['BusinessTravel'].classes_)
        DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
        Department = st.selectbox("Department", encoders['Department'].classes_)
        DistanceFromHome = st.slider("Distance From Home", 1, 30, 10)
        Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        EducationField = st.selectbox("Education Field", encoders['EducationField'].classes_)
        EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        Gender = st.selectbox("Gender", encoders['Gender'].classes_)

    with col2:
        HourlyRate = st.slider("Hourly Rate", 30, 100, 60)
        JobInvolvement = st.slider("Job Involvement", 1, 4, 3)
        JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        JobRole = st.selectbox("Job Role", encoders['JobRole'].classes_)
        JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        MaritalStatus = st.selectbox("Marital Status", encoders['MaritalStatus'].classes_)
        MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
        MonthlyRate = st.number_input("Monthly Rate", 1000, 25000, 10000)
        NumCompaniesWorked = st.slider("Num Companies Worked", 0, 10, 2)

    with col3:
        OverTime = st.selectbox("OverTime", encoders['OverTime'].classes_)
        PercentSalaryHike = st.slider("Percent Salary Hike", 10, 25, 15)
        PerformanceRating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
        TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 10, 2)
        WorkLifeBalance = st.slider("Work Life Balance", 1, 4, 3)
        YearsAtCompany = st.slider("Years At Company", 0, 40, 5)
        YearsInCurrentRole = st.slider("Years In Current Role", 0, 18, 5)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
        YearsWithCurrManager = st.slider("Years With Current Manager", 0, 17, 3)

    submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        # Feature Engineering
        EngagementScore = np.mean([JobInvolvement, EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance])
        if YearsAtCompany < 3:
            TenureCategory = 'Short'
        elif YearsAtCompany < 6:
            TenureCategory = 'Medium'
        else:
            TenureCategory = 'Long'

        # Encode all categorical variables using saved encoders
        def encode(val, col_name):
            return encoders[col_name].transform([val])[0]

        input_data = [
            Age,
            encode(BusinessTravel, 'BusinessTravel'),
            DailyRate,
            encode(Department, 'Department'),
            DistanceFromHome,
            Education,
            encode(EducationField, 'EducationField'),
            EnvironmentSatisfaction,
            encode(Gender, 'Gender'),
            HourlyRate,
            JobInvolvement,
            JobLevel,
            encode(JobRole, 'JobRole'),
            JobSatisfaction,
            encode(MaritalStatus, 'MaritalStatus'),
            MonthlyIncome,
            MonthlyRate,
            NumCompaniesWorked,
            encode(OverTime, 'OverTime'),
            PercentSalaryHike,
            PerformanceRating,
            RelationshipSatisfaction,
            StockOptionLevel,
            TotalWorkingYears,
            TrainingTimesLastYear,
            WorkLifeBalance,
            YearsAtCompany,
            YearsInCurrentRole,
            YearsSinceLastPromotion,
            YearsWithCurrManager,
            EngagementScore,
            encode(TenureCategory, 'TenureCategory') if 'TenureCategory' in encoders else {'Short': 0, 'Medium': 1, 'Long': 2}[TenureCategory]
        ]

        # Scale input
        input_scaled = scaler.transform([input_data])

        # Prediction
        prediction = model.predict(input_scaled)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
           st.success("✅ The employee is likely to **Stay** with the company.")
        else:
           st.error("⚠️ The employee is likely to **Leave** the company.")





#----------------------------------------------------------------------------------------------------------------------------------------------








