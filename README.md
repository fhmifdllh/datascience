Notebook **Project.ipynb** yang Anda rujuk berfokus pada klasifikasi dataset **Iris** menggunakan tiga algoritma: **Logistic Regression**, **Decision Tree**, dan **Random Forest**. Berikut adalah penjelasan dari isi kode dalam notebook tersebut:

1. **Import Library yang Diperlukan**
Notebook dimulai dengan mengimpor pustaka-pustaka penting seperti `pandas`, `numpy`, `matplotlib.pyplot`, dan `seaborn` untuk manipulasi data dan visualisasi. Selain itu, mengimpor modul dari `sklearn` untuk pemodelan machine learning.

2. **Memuat Dataset Iris**
Dataset Iris diambil dari `sklearn.datasets` dan dimuat ke dalam variabel `iris`. Data ini kemudian dikonversi menjadi DataFrame `pandas` untuk memudahkan manipulasi dan analisis.

3. **Eksplorasi Data**
Bagian ini mencakup analisis deskriptif seperti melihat lima baris pertama data, memeriksa informasi dataset, dan menghasilkan statistik deskriptif. Visualisasi data dilakukan menggunakan `seaborn.pairplot` untuk melihat hubungan antar fitur dan distribusi kelas.

4. **Persiapan Data untuk Pemodelan**
Fitur (X) dan label (y) dipisahkan. Data kemudian dibagi menjadi set pelatihan dan pengujian menggunakan `train_test_split` dengan proporsi 80% untuk pelatihan dan 20% untuk pengujian.

5. **Standarisasi Fitur**
Standarisasi dilakukan menggunakan `StandardScaler` untuk memastikan semua fitur memiliki skala yang sama, yang penting untuk algoritma seperti Logistic Regression.

6. **Pelatihan dan Evaluasi Model**
Tiga model dikembangkan dan dievaluasi:

- **Logistic Regression**: Model dilatih dan dievaluasi menggunakan metrik akurasi dan laporan klasifikasi.

- **Decision Tree**: Model dilatih dengan visualisasi pohon keputusan dan evaluasi performa.

- **Random Forest**: Model dilatih, dan pentingnya fitur dievaluasi serta visualisasi.

Setiap model dievaluasi menggunakan confusion matrix dan visualisasi untuk memahami performa klasifikasi.

7. **Kesimpulan**
Bagian akhir membandingkan kinerja ketiga model berdasarkan metrik evaluasi dan visualisasi, serta memberikan wawasan tentang model mana yang paling efektif untuk dataset ini.

Notebook ini menyediakan panduan komprehensif untuk memahami proses klasifikasi dataset Iris menggunakan berbagai algoritma machine learning, mulai dari eksplorasi data hingga evaluasi model. 