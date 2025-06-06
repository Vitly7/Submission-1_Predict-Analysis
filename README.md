
# Laporan Prediksi Harga Rumah Menggunakan Machine Learning - Vittorio Fiorentino

---

## Domain Proyek

Pergerakan harga properti atau rumah merupakan aspek penting dalam dunia real estat, baik bagi pembeli, penjual, maupun investor. Dalam era digital dan data-driven saat ini, prediksi harga rumah berbasis data historis menjadi pendekatan yang sangat diminati karena mampu memberikan wawasan yang lebih akurat dalam pengambilan keputusan jual beli atau investasi.

Dengan meningkatnya ketersediaan data properti (seperti ukuran rumah, lokasi, jumlah kamar, dan tahun pembangunan), serta kemajuan teknologi machine learning, analisis prediktif harga rumah menjadi solusi yang efektif dan efisien dalam memahami tren pasar properti.


---

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi harga rumah berdasarkan data historis dan karakteristik rumah?

2. Model machine learning mana yang memberikan performa terbaik dalam memprediksi harga rumah?

### Goals

1. Menghasilkan model prediktif yang mampu memperkirakan harga rumah berdasarkan fitur-fitur seperti luas bangunan, jumlah kamar, lokasi, kondisi, dan lainnya.

2. Membandingkan performa beberapa algoritma machine learning untuk menentukan model terbaik dalam memprediksi harga rumah.

### Solution Statement

Untuk mencapai tujuan di atas, digunakan tiga algoritma machine learning, yaitu:
- K-Nearest Neighbors (KNN)

Digunakan untuk memprediksi harga rumah berdasarkan kedekatan fitur rumah lain dengan harga yang telah diketahui.

- Random Forest (RF)

Model ensemble berbasis pohon keputusan yang dapat menangani data kompleks dan memberikan hasil prediksi yang stabil.

- Gradient Boosting

Model boosting yang secara bertahap memperbaiki kesalahan prediksi dari model sebelumnya untuk meningkatkan akurasi.

Evaluasi model dilakukan menggunakan Root Mean Squared Error (RMSE) pada data pelatihan dan pengujian.


---

## Data Understanding
Dataset diperoleh dari kaggle/Github (Sudah disimpan ke gdrive)= 'https://drive.google.com/uc?id=1JnHqcZDejRC0Bdei0VlpeiHyf9PvG3cz'
Dataset yang digunakan merupakan data rumah/properti historis dengan atribut:
- Jumlah baris = 21613 baris
- Jumlah kolom = 21 kolom
- Tidak ada missing Value
- Semua kolom bertipe Numeric (Kolom Date harusnya bertipe datetime)

---

### Exploratory Data Analysis (EDA)

Untuk memahami karakteristik dasar dari dataset rumah yang digunakan, dilakukan analisis statistik deskriptif terhadap fitur-fitur numerik.

- Memperkecil scope variabel pada dataset untuk menyederhanakan model dan menghindari overfitting.
- Menangani Outlier

**Handling Outliers**

![ss4](https://github.com/Vitly7/Submission-1_Predict-Analysis/blob/52f678e28e7d4acf54b01c077dd0a8779dccae23/gambar/boxplot.png)

Penjelasan BoxPlot:

- bathrooms Boxplot: Terlihat banyak outlier di atas (hingga 8 kamar mandi). Rumah dengan lebih dari 4–5 kamar mandi itu jarang dan bisa dikategorikan outlier.

- sqft_living Boxplot: Outlier ekstrem sangat banyak di sisi atas (hingga >13.000 sqft). Rumah dengan luas >8000 sqft adalah mansion dan sangat tidak umum, bisa mendistorsi model.

- floors Boxplot: Hampir tidak ada outlier. Rentang antara 1 hingga 3.5 lantai, distribusinya cukup stabil.

- bedrooms Boxplot: Outlier bisa terjadi untuk rumah dengan 33 kamar tidur.

- price Boxplot: Harga memang ada outliers. Mungkin karena ada rumah mahal/mansion


**Agregasi rata-rata harga berdasarkan kategori luas rumah**

![ss4](https://github.com/Vitly7/Submission-1_Predict-Analysis/blob/52f678e28e7d4acf54b01c077dd0a8779dccae23/gambar/agregasi.png)

Berdasarkan hasil visualisasi, berikut adalah beberapa insight yang bisa didapatkan:

Distribusi Harga Jual Berdasarkan Luas Rumah: Bar chart menunjukkan harga jual ('selling_price') untuk berbagai kategori luas rumah ('sqft_living'). Sumbu X adalah luas rumah dan sumbu Y adalah harga jual.

Kategori dengan Harga Jual Terendah: Range 0-1000

Kategori dengan Harga Jual Tertinggi: Range 4001+

Tren Peningkatan Harga Jual: Terlihat jelas ada tren peningkatan harga jual dari kiri ke kanan.

#### Multivariate Analysis

**Variabel Fitur Numerik yang dipakai untuk analisis**

Fitur numerik: ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'condition', 'grade', 'yr_built', 'yr_renovated']

![ss4](https://github.com/Vitly7/Submission-1_Predict-Analysis/blob/52f678e28e7d4acf54b01c077dd0a8779dccae23/gambar/correlation.png)

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

Reduksi dimensi berfungsi untuk mengurangi dimensi data, meningkatkan performa model, mempermudah visualisasi
Hasil reduksi ratio 
array([0.983, 0.017])


- Train Test Split: Membagi data menjadi data latih dan data uji dengan rasio 90:10
Total of sample in whole dataset: 21600

Total of sample in train dataset: 19440

Total of sample in test dataset: 2160

- Standarisasi Numeric

            bedrooms	bathrooms	    sqft_living	        floors	       yr_built	       grade        yr_renovated
count   1.728000e+04	1.728000e+04	1.728000e+04	1.728000e+04	1.728000e+04	1.728000e+04	1.728000e+04
mean	4.070818e-17	-6.517420e-17	-1.798972e-16	7.360367e-17	2.872599e-15	5.304399e-17	5.345518e-18
std	    1.000029e+00	1.000029e+00	1.000029e+00	1.000029e+00	1.000029e+00	1.000029e+00	1.000029e+00
min	    -2.183627e+00	-2.036927e+00	-2.007164e+00	-9.080882e-01	-2.419629e+00	-3.960760e+00	-2.147758e-01
25% 	-4.251651e-01	-8.240421e-01	-7.578735e-01	-9.080882e-01	-6.788543e-01	-5.557311e-01	-2.147758e-01
50%	    -4.251651e-01	2.155739e-01	-1.748714e-01	1.713374e-02	1.403338e-01	-5.557311e-01	-2.147758e-01
75%	    7.471426e-01	5.621126e-01	5.866007e-01	9.423557e-01	8.912561e-01	2.955262e-01	-2.147758e-01
max	    2.505604e+00	2.121537e+00	2.586804e+00	3.718022e+00	1.505647e+00	4.551812e+00	4.701859e+00

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


**Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1**

X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

---

## Evaluation

### Metrik Evaluasi

![ss4](https://github.com/Vitly7/Submission-1_Predict-Analysis/blob/52f678e28e7d4acf54b01c077dd0a8779dccae23/gambar/hasil.png)
Penjelasan:

- Boosting ~5.4 ~4.4 Overfitting terlihat → error train & test cukup besar tapi tidak seimbang. Boosting cenderung fit terlalu dalam ke data latih.
- KNN ~0.26 ~0.22 Cukup stabil → train dan test error mirip → model tidak overfit atau underfit.
- RF ~0.04 ~0.2 Sangat baik → performa sangat bagus di train dan cukup baik di test. Hampir tidak overfit.

### Hasil Pengujian Data

|    y_true     | prediksi_KNN   | prediksi_RF  | prediksi_Boosting |
| 	            |                |              |                   |   
|   270000.0    |   318473.2     |   320560.6   |    382082.5       |


**Evaluasi Akurasi Prediksi:**

Hasil Asli = 270000

- KNN 334973 -> Cukup dekat
- RandomForest 324502.9 -> Paling dekat
- Boosting 383839.0 -> Cukup jauh dari aslinya

---

### Kesimpulan:
- Semua model cukup baik kali ini, terutama Random Forest yang prediksinya hampir identik dengan nilai sebenarnya.

- Model Boosting cenderung underestimate (meremehkan nilai).

- KNN juga cukup akurat, hanya sedikit lebih tinggi.
