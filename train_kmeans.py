import os
import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =====================================================
# MEMBACA DATASET
# =====================================================

dataset_path = os.path.join("Dataset", "dataset.csv")

df = pd.read_csv(dataset_path)

print("=" * 50)
print("5 DATA PERTAMA")
print("=" * 50)
print(df.head())

print("\nNama Kolom Sebelum Perbaikan:")
print(df.columns.tolist())

# =====================================================
# PERBAIKAN NAMA KOLOM
# =====================================================

# Menghapus spasi pada nama kolom
df.columns = df.columns.str.strip()

# Jika kolom pertama bukan x (misalnya menjadi cd ..x)
if df.columns[0] != "x":
    print("\nKolom pertama diperbaiki menjadi 'x'")
    df.rename(columns={df.columns[0]: "x"}, inplace=True)

print("\nNama Kolom Setelah Perbaikan:")
print(df.columns.tolist())

# =====================================================
# MEMILIH FITUR
# =====================================================

X = df[
    [
        "x",
        "y",
        "population_density",
        "traffic_flow",
        "competitor_count",
        "is_commercial",
    ]
]

# =====================================================
# NORMALISASI
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# MEMBUAT MODEL KMEANS
# =====================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster

# =====================================================
# EVALUASI
# =====================================================

score = silhouette_score(X_scaled, cluster)

print("\n" + "=" * 50)
print("HASIL EVALUASI")
print("=" * 50)
print("Silhouette Score :", round(score, 4))

# =====================================================
# MEMBUAT FOLDER MODELS
# =====================================================

os.makedirs("models", exist_ok=True)

joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(scaler, "models/kmeans_scaler.pkl")

# =====================================================
# MENYIMPAN HASIL CLUSTER
# =====================================================

hasil_path = os.path.join("Dataset", "hasil_cluster.csv")

df.to_csv(hasil_path, index=False)

# =====================================================
# INFORMASI HASIL
# =====================================================

print("\n" + "=" * 50)
print("JUMLAH DATA SETIAP CLUSTER")
print("=" * 50)
print(df["Cluster"].value_counts().sort_index())

print("\n" + "=" * 50)
print("SEMUA PROSES BERHASIL")
print("=" * 50)

print("File yang dihasilkan:")
print("models/kmeans.pkl")
print("models/kmeans_scaler.pkl")
print("Dataset/hasil_cluster.csv")