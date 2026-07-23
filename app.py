import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="UAS Data Mining",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# LOAD MODEL & DATA DENGAN CACHING (OPTIMASI UTAMA)
# =====================================================

@st.cache_resource
def load_models():
    knn = joblib.load("models/knn.pkl")
    nb = joblib.load("models/naive_bayes.pkl")
    dt = joblib.load("models/decision_tree.pkl")
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans.pkl")
    kmeans_scaler = joblib.load("models/kmeans_scaler.pkl")
    return knn, nb, dt, scaler, kmeans, kmeans_scaler

@st.cache_data
def load_data():
    diabetes = pd.read_csv("Dataset/diabetes.csv")
    cluster = pd.read_csv("Dataset/hasil_cluster.csv")
    return diabetes, cluster

# Panggil fungsi cache
knn, nb, dt, scaler, kmeans, kmeans_scaler = load_models()
diabetes, cluster = load_data()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📚 UAS DATA MINING")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Dashboard",
        "🩺 Prediksi Diabetes",
        "☕ Clustering Gerai Kopi",
        "📊 Evaluasi Model",
        "ℹ️ Tentang"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if menu == "🏠 Dashboard":

    st.title("🩺 Implementasi Supervised dan Unsupervised Learning")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Data Diabetes", len(diabetes))
    col2.metric("Jumlah Data Gerai", len(cluster))
    col3.metric("Jumlah Cluster", cluster["Cluster"].nunique())

    st.markdown("---")
    st.subheader("Dataset Diabetes")
    st.dataframe(diabetes.head(), use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset Gerai Kopi")
    st.dataframe(cluster.head(), use_container_width=True)

    st.markdown("---")
    st.success(
        """
        Aplikasi ini merupakan implementasi
        Supervised Learning dan Unsupervised Learning.
        ✔ KNN | ✔ Naive Bayes | ✔ Decision Tree | ✔ K-Means Clustering
        """
    )

# =====================================================
# PREDIKSI DIABETES
# =====================================================

elif menu == "🩺 Prediksi Diabetes":

    st.title("🩺 Prediksi Risiko Diabetes")
    st.markdown("Masukkan data pasien kemudian pilih algoritma yang ingin digunakan.")
    st.markdown("---")

    model = st.selectbox(
        "Pilih Algoritma",
        ("KNN", "Naive Bayes", "Decision Tree")
    )

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
        pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.50)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    st.markdown("---")

    if st.button("Prediksi Diabetes"):
        data = np.array([[
            pregnancies, glucose, blood, skin,
            insulin, bmi, pedigree, age
        ]])

        data = scaler.transform(data)

        if model == "KNN":
            hasil = knn.predict(data)[0]
        elif model == "Naive Bayes":
            hasil = nb.predict(data)[0]
        else:
            hasil = dt.predict(data)[0]

        st.markdown("---")
        st.subheader("Hasil Prediksi")
        st.write("Model yang digunakan :", model)

        if hasil == 1:
            st.error("⚠️ Pasien diprediksi MENGIDAP DIABETES.")
        else:
            st.success("✅ Pasien diprediksi TIDAK MENGIDAP DIABETES.")

        st.markdown("---")
        st.subheader("Data Input")

        hasil_df = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood],
            "SkinThickness": [skin],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [pedigree],
            "Age": [age]
        })

        st.dataframe(hasil_df, use_container_width=True)
        st.info("Prediksi menggunakan model Machine Learning Pima Indians Diabetes.")

# =====================================================
# EVALUASI MODEL
# =====================================================

elif menu == "📊 Evaluasi Model":

    st.title("📊 Evaluasi Model Machine Learning")

    X = diabetes.drop("Outcome", axis=1)
    y = diabetes["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test = scaler.transform(X_test)

    pred_knn = knn.predict(X_test)
    pred_nb = nb.predict(X_test)
    pred_dt = dt.predict(X_test)

    hasil = pd.DataFrame({
        "Model": ["KNN", "Naive Bayes", "Decision Tree"],
        "Accuracy": [
            round(accuracy_score(y_test, pred_knn), 4),
            round(accuracy_score(y_test, pred_nb), 4),
            round(accuracy_score(y_test, pred_dt), 4)
        ],
        "Precision": [
            round(precision_score(y_test, pred_knn), 4),
            round(precision_score(y_test, pred_nb), 4),
            round(precision_score(y_test, pred_dt), 4)
        ],
        "Recall": [
            round(recall_score(y_test, pred_knn), 4),
            round(recall_score(y_test, pred_nb), 4),
            round(recall_score(y_test, pred_dt), 4)
        ],
        "F1-Score": [
            round(f1_score(y_test, pred_knn), 4),
            round(f1_score(y_test, pred_nb), 4),
            round(f1_score(y_test, pred_dt), 4)
        ]
    })

    st.subheader("Perbandingan Model")
    st.dataframe(hasil, use_container_width=True)

    st.markdown("---")
    st.subheader("Grafik Accuracy")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(hasil["Model"], hasil["Accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

    st.markdown("---")
    pilihan = st.selectbox("Pilih Model", ("KNN", "Naive Bayes", "Decision Tree"))

    if pilihan == "KNN":
        cm = confusion_matrix(y_test, pred_knn)
    elif pilihan == "Naive Bayes":
        cm = confusion_matrix(y_test, pred_nb)
    else:
        cm = confusion_matrix(y_test, pred_dt)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax2)
    st.pyplot(fig2)

    st.success("Evaluasi model berhasil ditampilkan.")

# =====================================================
# CLUSTERING GERAI KOPI
# =====================================================

elif menu == "☕ Clustering Gerai Kopi":

    st.title("☕ Analisis Clustering Gerai Kopi")
    st.write("Halaman ini menganalisis persebaran gerai kopi menggunakan K-Means.")
    st.markdown("---")

    st.subheader("Dataset Hasil Clustering")
    st.dataframe(cluster.head(), use_container_width=True)

    st.markdown("---")
    st.subheader("Visualisasi Cluster")

    fig, ax = plt.subplots(figsize=(8, 6))
    warna = ["red", "blue", "green", "orange", "purple"]

    for i in sorted(cluster["Cluster"].unique()):
        data_cluster = cluster[cluster["Cluster"] == i]
        ax.scatter(
            data_cluster["x"],
            data_cluster["y"],
            s=50,
            color=warna[i % len(warna)],
            label=f"Cluster {i}"
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Hasil Clustering Gerai Kopi")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Prediksi Lokasi Baru")

    col1, col2 = st.columns(2)

    with col1:
        x = st.number_input("Koordinat X", value=50.0)
        y = st.number_input("Koordinat Y", value=50.0)
        population = st.number_input("Population Density", value=3000.0)

    with col2:
        traffic = st.number_input("Traffic Flow", value=700.0)
        competitor = st.number_input("Competitor Count", value=3)
        commercial = st.selectbox("Commercial Area", [0, 1])

    if st.button("Prediksi Cluster"):
        data_baru = np.array([[
            x, y, population, traffic, competitor, commercial
        ]])

        data_baru = kmeans_scaler.transform(data_baru)
        hasil = kmeans.predict(data_baru)[0]

        st.success(f"Lokasi termasuk Cluster {hasil}")

        # Mencari kolom traffic secara dinamis agar aman dari error KeyError
        traffic_col = next((col for col in cluster.columns if 'traffic' in col.lower()), None)
        
        if traffic_col:
            rata = cluster.groupby("Cluster")[traffic_col].mean()
            cluster_sepi = rata.idxmin()

            if hasil == cluster_sepi:
                st.error("⚠️ Lokasi diprediksi termasuk Zona Sepi")
            else:
                st.success("✅ Lokasi diprediksi termasuk Zona Ramai")

    st.markdown("---")
    st.subheader("Jumlah Gerai per Cluster")
    jumlah = cluster["Cluster"].value_counts().sort_index()
    st.bar_chart(jumlah)

# =====================================================
# TENTANG
# =====================================================

elif menu == "ℹ️ Tentang":

    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    ## Implementasi Supervised dan Unsupervised Learning

    **Mata Kuliah**
    Data Mining

    **Algoritma Klasifikasi**
    - K-Nearest Neighbor
    - Naive Bayes
    - Decision Tree

    **Algoritma Clustering**
    - K-Means

    **Dataset**
    - Pima Indians Diabetes
    - Coffee Shop Location Dataset

    **Tools**
    - Python
    - Streamlit
    - Scikit-Learn
    - Pandas
    - Matplotlib

    Aplikasi ini dibuat untuk memenuhi tugas **Ujian Akhir Semester (UAS) Data Mining**.
    """)

    st.success("Terima kasih telah menggunakan aplikasi ini.")
