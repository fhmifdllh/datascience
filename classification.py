import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
iris = datasets.load_iris()
X = iris.data  # Fitur
y = iris.target  # Label

# 2. Membagi Data menjadi Training dan Testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalisasi Data (Standarisasi)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Inisialisasi dan Melatih Model SVM
svm = SVC(kernel='linear')  # Kernel bisa diubah menjadi 'rbf', 'poly', atau 'sigmoid'
svm.fit(X_train, y_train)

# 5. Prediksi Data Uji
y_pred = svm.predict(X_test)

# 6. Evaluasi Model
print("Akurasi Model:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# 7. Menampilkan Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()