import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report
)

# ===============================
# 1. PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ===============================
# 2. CUSTOM CSS 
# ===============================
st.markdown("""
<style>

body { background-color: #F2F4F4; }

.sidebar .sidebar-content {
    background-color: #283747;
}

.main-title {
    font-size: 40px;
    font-weight: 900;
    color: #2C3E50;
    text-align: center;
    margin-bottom: 10px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #34495E;
    margin-top: 20px;
}

.metric-box {
    background-color: #FDFEFE;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #D5D8DC;
}

.support-box {
    background-color: #EBF5FB;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #AED6F1;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# 3. SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Home", "EDA", "Modeling", "Feature Importance"]
)

# ===============================
# LOAD DATASET
# ===============================
file_path = "D:/UAS Data Mining_Wega ramadhan_Rizky Aditiya S/UAS_Data_Mining_Wega_Rizy.csv"
df = pd.read_csv(file_path)


# ===============================
# HOME PAGE
# ===============================
if menu == "Home":
    st.markdown('<div class="main-title">Klasifikasi Penyakit Jantung Menggunakan XGBoost</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dataset</div>', unsafe_allow_html=True)
    st.dataframe(df.head())



# ===============================
# EDA PAGE
# ===============================
elif menu == "EDA":
    st.markdown('<div class="main-title">📊 Visualisasi Dataset (EDA)</div>',
                unsafe_allow_html=True)

    # ---------- Distribusi Target ----------
    st.markdown('<div class="section-title">Distribusi Kelas Target</div>',
                unsafe_allow_html=True)

    fig1, ax1 = plt.subplots()
    sns.countplot(x=df["target"], palette="viridis", ax=ax1)
    st.pyplot(fig1)

    # ---------- Heatmap ----------
    st.markdown('<div class="section-title">Heatmap Korelasi</div>',
                unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(df.corr(), annot=False, cmap="viridis", ax=ax2)
    st.pyplot(fig2)


# ===============================
# MODEL TRAINING & EVALUATION PAGE
# ===============================
elif menu == "Modeling":

    st.markdown('<div class="main-title">⚙️ Modeling & Evaluasi XGBoost</div>',
                unsafe_allow_html=True)

    # PREPROCESSING
    df_clean = df.drop_duplicates().dropna()

    X = df_clean.drop("target", axis=1)
    y = df_clean["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # MODEL
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # METRICS
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Hasil Evaluasi</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-box">
                <b>Accuracy:</b> {acc}<br>
                <b>Precision:</b> {prec}<br>
                <b>Recall:</b> {rec}<br>
                <b>F1-Score:</b> {f1}<br>
                <b>ROC-AUC:</b> {auc}
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Support</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="support-box">
                Kelas 0: <b>{(y_test==0).sum()}</b><br>
                Kelas 1: <b>{(y_test==1).sum()}</b><br>
                Total Data Uji: <b>{len(y_test)}</b>
            </div>
        """, unsafe_allow_html=True)

    # CONFUSION MATRIX
    st.markdown('<div class="section-title">Confusion Matrix</div>',
                unsafe_allow_html=True)

    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax3)
    st.pyplot(fig3)

    # ROC CURVE
    st.markdown('<div class="section-title">ROC Curve</div>',
                unsafe_allow_html=True)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig4, ax4 = plt.subplots()
    ax4.plot(fpr, tpr, label=f"AUC = {auc}")
    ax4.plot([0, 1], [0, 1], linestyle="--", color="gray")
    st.pyplot(fig4)


# ===============================
# FEATURE IMPORTANCE PAGE
# ===============================
elif menu == "Feature Importance":

    st.markdown('<div class="main-title">🔍 Feature Importance</div>',
                unsafe_allow_html=True)

    df_clean = df.drop_duplicates().dropna()
    X = df_clean.drop("target", axis=1)
    y = df_clean["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier()
    model.fit(X_scaled, y)

    importance = model.feature_importances_

    fig5, ax5 = plt.subplots()
    sns.barplot(x=importance, y=X.columns, palette="viridis", ax=ax5)
    ax5.set_title("Pentingnya Fitur")
    st.pyplot(fig5)
