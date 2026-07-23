# UAS Data Mining

## Implementasi Supervised Learning dan Unsupervised Learning Menggunakan Streamlit

### Deskripsi

Aplikasi ini merupakan implementasi algoritma Machine Learning untuk memenuhi tugas Ujian Akhir Semester (UAS) mata kuliah Data Mining.

Aplikasi terdiri dari dua bagian utama yaitu:

1. Prediksi Diabetes (Supervised Learning)
2. Clustering Gerai Kopi (Unsupervised Learning)

---

## Algoritma yang Digunakan

### Supervised Learning

- K-Nearest Neighbor (KNN)
- Naive Bayes
- Decision Tree

### Unsupervised Learning

- K-Means Clustering

---

## Dataset

### Dataset Diabetes

Pima Indians Diabetes Dataset

Target:

- Outcome

Fitur:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

### Dataset Gerai Kopi

Coffee Shop Location Dataset

Fitur:

- x
- y
- population_density
- traffic_flow
- competitor_count
- is_commercial

---

## Tools

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Joblib

---

## Cara Menjalankan

Install library

```bash
pip install -r requirements.txt
```

Menjalankan aplikasi

```bash
streamlit run app.py
```

---

## Struktur Project

```
UAS-DataMining
│
├── app.py
├── train_diabetes.py
├── train_kmeans.py
├── requirements.txt
├── README.md
│
├── Dataset
│   ├── diabetes.csv
│   ├── dataset.csv
│   └── hasil_cluster.csv
│
└── models
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── decision_tree.pkl
    ├── scaler.pkl
    ├── kmeans.pkl
    └── kmeans_scaler.pkl
```

---

## Author

Nama : Syah Arifka Fizirqi

NIM : 23146077

Program Studi : Sistem Informasi

Universitas : Abulyatama aceh
