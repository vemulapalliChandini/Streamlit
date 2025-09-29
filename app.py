# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="📊 Loan Repayment Prediction Dashboard",
    layout="wide"
)

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "KPIs", "Model Errors", "Deep Analysis"])

# ----------------------------
# Load / Generate Dataset
# ----------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500
    data = {
        "Meets_Credit_Policy": np.random.choice([0, 1], size=n, p=[0.3, 0.7]),
        "Loan_Purpose": np.random.choice(
            ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business"],
            size=n
        ),
        "Interest_Rate": np.round(np.random.uniform(5, 25, n), 2),
        "Monthly_Installment": np.round(np.random.uniform(100, 2000, n), 2),
        "Log_Annual_Income": np.round(np.log(np.random.randint(20000, 200000, n)), 2),
        "Debt_To_Income_Ratio": np.round(np.random.uniform(0, 40, n), 2),
        "FICO_Score": np.random.randint(600, 850, n),
        "Credit_History_Length": np.random.randint(100, 5000, n),
        "Revolving_Balance": np.random.randint(0, 50000, n),
        "Revolving_Utilization": np.round(np.random.uniform(0, 100, n), 2),
        "Recent_Inquiries": np.random.randint(0, 10, n),
        "Delinquencies_2yrs": np.random.randint(0, 5, n),
        "Public_Records": np.random.randint(0, 3, n),
        "Loan_Status": np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

df = load_data()

# ----------------------------
# Train a Simple Model
# ----------------------------
X = df.drop(columns=["Loan_Status", "Loan_Purpose"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------
# Page Content
# ----------------------------

if page == "Home":
    st.title("📊 Loan Repayment Prediction Dashboard")
    st.write("Welcome! This dashboard demonstrates a machine learning model for predicting loan repayment.")
    st.dataframe(df.head(10))

elif page == "KPIs":
    st.title("📌 Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Loans", len(df))
    col2.metric("Defaults Predicted", int(y_pred.sum()))
    col3.metric("Average FICO Score", int(df["FICO_Score"].mean()))

    st.subheader("📈 Loan Purpose Distribution")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(data=df, x="Loan_Purpose", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif page == "Model Errors":
    st.title("❌ Model Errors & Performance")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Paid", "Default"], yticklabels=["Paid", "Default"], ax=ax)
    st.pyplot(fig)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "r--")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

elif page == "Deep Analysis":
    st.title("🔎 Deep Analysis")
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("FICO Score vs Default Probability")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["FICO_Score"], y=df["Loan_Status"], alpha=0.6)
    st.pyplot(fig)
