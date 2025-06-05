
# Laporan Prediksi Harga Rumah Menggunakan Machine Learning - Vittorio Fiorentino

---

## Domain Proyek

Pergerakan harga saham menjadi aspek penting dalam dunia keuangan, baik bagi investor individu maupun institusional. Dalam era digital saat ini, prediksi harga saham berbasis data menjadi semakin diminati karena mampu memberikan insight untuk pengambilan keputusan yang lebih akurat. Dengan meningkatnya volume data keuangan dan kemampuan komputasi, pemanfaatan machine learning untuk menganalisis pola harga saham menjadi solusi potensial yang efektif dan efisien.

Menurut [Fischer & Krauss, 2018], model deep learning seperti LSTM dapat mengalahkan model tradisional dalam prediksi harga saham harian. Hal ini memperkuat urgensi dan relevansi topik ini dalam konteks bisnis dan teknologi modern.


---

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi harga penutupan saham berdasarkan data historis harga saham?
2. Model machine learning mana yang memberikan performa terbaik dalam memprediksi harga saham?

### Goals

1. Menghasilkan model yang mampu memprediksi harga saham berdasarkan fitur-fitur seperti Open, High, Low, dan Volume.
2. Membandingkan beberapa algoritma machine learning untuk mengetahui performa terbaik.

### Solution Statement

Untuk mencapai tujuan di atas, digunakan tiga algoritma machine learning, yaitu:
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Gradient Boosting

Evaluasi model dilakukan menggunakan Root Mean Squared Error (RMSE) pada data pelatihan dan pengujian.


---

## Data Understanding
Dataset diperoleh dari kaggle/Github (Sudah disimpan ke gdrive)= 'https://drive.google.com/uc?id=1JnHqcZDejRC0Bdei0VlpeiHyf9PvG3cz'
Dataset yang digunakan merupakan data harga saham historis dengan atribut:
- Jumlah baris = 21613 baris
- Jumlah kolom = 21 kolom
- Tidak ada missing Value
- Semua kolom bertipe Numeric (Kolom Date harusnya bertipe datetime)

---

### Exploratory Data Analysis (EDA)

Untuk memahami karakteristik dasar dari dataset harga saham yang digunakan, dilakukan analisis statistik deskriptif terhadap fitur-fitur numerik.

- Memperkecil scope variabel pada dataset untuk menyederhanakan model dan menghindari overfitting.
- Menangani Outlier

Penjelasan BoxPlot:

- bathrooms Boxplot: Terlihat banyak outlier di atas (hingga 8 kamar mandi). Rumah dengan lebih dari 4–5 kamar mandi itu jarang dan bisa dikategorikan outlier.

- sqft_living Boxplot: Outlier ekstrem sangat banyak di sisi atas (hingga >13.000 sqft). Rumah dengan luas >8000 sqft adalah mansion dan sangat tidak umum, bisa mendistorsi model.

- floors Boxplot: Hampir tidak ada outlier. Rentang antara 1 hingga 3.5 lantai, distribusinya cukup stabil.

- bedrooms Boxplot: Outlier bisa terjadi untuk rumah dengan 33 kamar tidur.

- price Boxplot: Harga memang ada outliers. Mungkin karena ada rumah mahal/mansion

Penjelasan Correlation Matrix:

🔺 grade (0.67) dan sqft_living (0.65) punya korelasi paling kuat dengan price → makin tinggi kualitas & luas rumah, makin mahal harganya.

🛁 bathrooms (0.48) dan 🛏️ bedrooms (0.32) juga cukup berpengaruh terhadap price, tapi tidak sekuat grade.

🚫 condition (0.04) dan yr_built (0.05) hampir tidak berkorelasi dengan price → usia atau kondisi rumah tidak terlalu memengaruhi harga.

💡 Korelasi antar fitur juga terlihat kuat antara:

- bathrooms & sqft_living (0.75)

- grade & sqft_living (0.76)

---

## Data Preparation

- Reduksi dimensi dengan PCA
- Train Test Split: Membagi data menjadi data latih dan data uji dengan rasio 90:10
Total of sample in whole dataset: 21600

Total of sample in train dataset: 19440

Total of sample in test dataset: 2160

- Standarisasi Numeric

---

## Modeling

Tiga model machine learning yang digunakan:
- **KNN**: Algoritma sederhana berbasis tetangga terdekat
- **Random Forest**: Model ensambel berbasis pohon keputusan
- **Gradient Boosting**: Model boosting yang menggabungkan banyak pohon lemah secara iteratif

| Model     |      Train         |       Test       |
|-----------|--------------------|------------------|
| KNN       | 39373913.354123	 | 48532713.700962  |
| RF        | 12473741.304761	 | 42857536.442852  |  
| Boosting  | 52643899.971604	 | 57222556.944881  |



**Analisis:**

- Boosting ~5.4 ~4.4 Overfitting terlihat → error train & test cukup besar tapi tidak seimbang. Boosting cenderung fit terlalu dalam ke data latih.
- KNN ~0.26 ~0.22 Cukup stabil → train dan test error mirip → model tidak overfit atau underfit.
- RF ~0.04 ~0.2 Sangat baik → performa sangat bagus di train dan cukup baik di test. Hampir tidak overfit.

---

## Evaluation

### Metrik Evaluasi
Penjelasan:

- Boosting ~5.4 ~4.4 Overfitting terlihat → error train & test cukup besar tapi tidak seimbang. Boosting cenderung fit terlalu dalam ke data latih.
- KNN ~0.26 ~0.22 Cukup stabil → train dan test error mirip → model tidak overfit atau underfit.
- RF ~0.04 ~0.2 Sangat baik → performa sangat bagus di train dan cukup baik di test. Hampir tidak overfit.

### Hasil Pengujian Data

|    y_true     | prediksi_KNN   | prediksi_RF  | prediksi_Boosting |
| 	            |                |              |                   |   
|   270000.0    |   318473.2     |   320560.6   |    382082.5       |

Evaluasi Akurasi Prediksi:

Hasil Asli = 270000

- KNN 334973 -> Cukup dekat
- RandomForest 324502.9 -> Paling dekat
- Boosting 383839.0 -> Cukup jauh dari aslinya

---

### Kesimpulan:
- Semua model cukup baik kali ini, terutama Random Forest yang prediksinya hampir identik dengan nilai sebenarnya.

- Model Boosting cenderung underestimate (meremehkan nilai).

- KNN juga cukup akurat, hanya sedikit lebih tinggi.
