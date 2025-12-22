import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

sns.set(style="whitegrid")

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Home", "EDA", "Modeling", "Feature Importance"]
)

# =====================================================
# LOAD DATASET (SAMA DENGAN IPYNB)
# =====================================================
df = pd.read_csv(
    r"D:\UAS Data Mining_Wega ramadhan_Rizky Aditiya S\UAS_Data_Mining_Wega_Rizy.csv"
)

# =====================================================
# HOME
# =====================================================
if menu == "Home":
    st.title("❤️ Klasifikasi Penyakit Jantung")
    st.write("Menggunakan **XGBoost** dan **Decision Tree**")

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    st.markdown("""
    **Target:**
    - **0** → Tidak Berisiko Penyakit Jantung  
    - **1** → Berisiko Penyakit Jantung
    """)

# =====================================================
# EDA
# =====================================================
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")

    # Distribusi Target
    st.subheader("Distribusi Kelas Target")
    fig1, ax1 = plt.subplots()
    sns.countplot(x=df["target"], ax=ax1)
    plt.title("Distribusi Kelas Target (0 = Tidak Berisiko, 1 = Berisiko)")
    ax1.set_xlabel("Target")
    ax1.set_ylabel("Jumlah")
    st.pyplot(fig1)

    # Heatmap Korelasi
    st.subheader("Heatmap Korelasi")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False, ax=ax2)
    st.pyplot(fig2)

# =====================================================
# MODELING
# =====================================================
elif menu == "Modeling":
    st.title("⚙️ Modeling & Evaluasi")

    # ---------------- Preprocessing ----------------
    df_clean = df.drop_duplicates().dropna()

    X = df_clean.drop("target", axis=1)
    y = df_clean["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        train_size=0.8,
        random_state=42
    )

    # ---------------- Model XGBoost ----------------
    model_xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model_xgb.fit(X_train, y_train)
    pred_xgb = model_xgb.predict(X_test)
    prob_xgb = model_xgb.predict_proba(X_test)[:, 1]

    # ---------------- Model Decision Tree ----------------
    model_dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=42
    )

    model_dt.fit(X_train, y_train)
    pred_dt = model_dt.predict(X_test)
    prob_dt = model_dt.predict_proba(X_test)[:, 1]

    # ---------------- Evaluasi ----------------
    def evaluate_model(y_true, y_pred, y_prob):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
            "ROC-AUC": roc_auc_score(y_true, y_prob)
        }

    xgb_result = evaluate_model(y_test, pred_xgb, prob_xgb)
    dt_result = evaluate_model(y_test, pred_dt, prob_dt)

    comparison = pd.DataFrame(
        [xgb_result, dt_result],
        index=["XGBoost", "Decision Tree"]
    )

    st.subheader("Tabel Perbandingan Model")
    st.dataframe(comparison)

    # ---------------- Confusion Matrix ----------------
    st.subheader("Confusion Matrix")

    col1, col2 = st.columns(2)

    with col1:
        st.write("XGBoost")
        fig3, ax3 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, pred_xgb),
                    annot=True, fmt="d", cmap="Blues", ax=ax3)
        st.pyplot(fig3)

    with col2:
        st.write("Decision Tree")
        fig4, ax4 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, pred_dt),
                    annot=True, fmt="d", cmap="Greens", ax=ax4)
        st.pyplot(fig4)

    # ---------------- ROC Curve ----------------
    st.subheader("ROC Curve")

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, prob_xgb)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, prob_dt)

    fig5, ax5 = plt.subplots()
    ax5.plot(fpr_xgb, tpr_xgb,
             label=f"XGBoost (AUC = {xgb_result['ROC-AUC']:.3f})")
    ax5.plot(fpr_dt, tpr_dt,
             label=f"Decision Tree (AUC = {dt_result['ROC-AUC']:.3f})")
    ax5.plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax5.set_xlabel("False Positive Rate")
    ax5.set_ylabel("True Positive Rate")
    ax5.legend()
    st.pyplot(fig5)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
elif menu == "Feature Importance":
    st.title("🔍 Feature Importance - XGBoost")

    df_clean = df.drop_duplicates().dropna()
    X = df_clean.drop("target", axis=1)
    y = df_clean["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model_xgb.fit(X_scaled, y)

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.barh(X.columns, model_xgb.feature_importances_)
    ax6.set_title("Feature Importance - XGBoost")
    st.pyplot(fig6)
