import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# ======================================================
# CEK LOKASI PROJECT
# ======================================================

print("=" * 50)
print("LOKASI PROJECT")
print("=" * 50)

print("Current Working Directory :")
print(os.getcwd())

print("\nIsi Folder Saat Ini :")
print(os.listdir())

print("\n")

# ======================================================
# MENENTUKAN PATH DATASET
# ======================================================

dataset_path = os.path.join(os.getcwd(), "Dataset", "diabetes.csv")

print("Lokasi Dataset :")
print(dataset_path)

if not os.path.exists(dataset_path):
    print("\nERROR!")
    print("File diabetes.csv tidak ditemukan.")

    print("\nPastikan struktur folder seperti berikut:\n")

    print(r"""
D:\UAS-DataMining
│
├── Dataset
│   ├── diabetes.csv
│   └── dataset.csv
│
├── train_diabetes.py
├── train_kmeans.py
├── app.py
└── models
""")

    print("\nIsi Folder Dataset:")

    dataset_folder = os.path.join(os.getcwd(), "Dataset")

    if os.path.exists(dataset_folder):
        print(os.listdir(dataset_folder))
    else:
        print("Folder Dataset tidak ditemukan!")

    exit()

# ======================================================
# MEMBACA DATASET
# ======================================================

df = pd.read_csv(dataset_path)

print("\n")
print("=" * 50)
print("5 DATA PERTAMA")
print("=" * 50)
print(df.head())

# ======================================================
# MEMISAHKAN FITUR DAN TARGET
# ======================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ======================================================
# SPLIT DATA
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================================
# NORMALISASI
# ======================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# MEMBUAT FOLDER MODELS
# ======================================================

os.makedirs("models", exist_ok=True)

# ======================================================
# MODEL KNN
# ======================================================

print("\n")
print("=" * 50)
print("MODEL KNN")
print("=" * 50)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred_knn))
print("Precision:", precision_score(y_test, pred_knn))
print("Recall   :", recall_score(y_test, pred_knn))
print("F1 Score :", f1_score(y_test, pred_knn))

joblib.dump(knn, "models/knn.pkl")

# ======================================================
# MODEL NAIVE BAYES
# ======================================================

print("\n")
print("=" * 50)
print("MODEL NAIVE BAYES")
print("=" * 50)

nb = GaussianNB()

nb.fit(X_train, y_train)

pred_nb = nb.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred_nb))
print("Precision:", precision_score(y_test, pred_nb))
print("Recall   :", recall_score(y_test, pred_nb))
print("F1 Score :", f1_score(y_test, pred_nb))

joblib.dump(nb, "models/naive_bayes.pkl")

# ======================================================
# MODEL DECISION TREE
# ======================================================

print("\n")
print("=" * 50)
print("MODEL DECISION TREE")
print("=" * 50)

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred_dt))
print("Precision:", precision_score(y_test, pred_dt))
print("Recall   :", recall_score(y_test, pred_dt))
print("F1 Score :", f1_score(y_test, pred_dt))

joblib.dump(dt, "models/decision_tree.pkl")

# ======================================================
# SIMPAN SCALER
# ======================================================

joblib.dump(scaler, "models/scaler.pkl")

# ======================================================
# SELESAI
# ======================================================

print("\n")
print("=" * 50)
print("SEMUA MODEL BERHASIL DIBUAT")
print("=" * 50)

print("File yang dihasilkan:")

print("models/knn.pkl")
print("models/naive_bayes.pkl")
print("models/decision_tree.pkl")
print("models/scaler.pkl")