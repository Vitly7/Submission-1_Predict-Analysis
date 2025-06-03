
# Prediksi Harga Saham Menggunakan Machine Learning

## Domain Proyek

Pergerakan harga saham menjadi aspek penting dalam dunia keuangan, baik bagi investor individu maupun institusional. Dalam era digital saat ini, prediksi harga saham berbasis data menjadi semakin diminati karena mampu memberikan insight untuk pengambilan keputusan yang lebih akurat. Dengan meningkatnya volume data keuangan dan kemampuan komputasi, pemanfaatan machine learning untuk menganalisis pola harga saham menjadi solusi potensial yang efektif dan efisien.

Menurut [Fischer & Krauss, 2018], model deep learning seperti LSTM dapat mengalahkan model tradisional dalam prediksi harga saham harian. Hal ini memperkuat urgensi dan relevansi topik ini dalam konteks bisnis dan teknologi modern.

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

## Data Understanding

Dataset yang digunakan merupakan data harga saham historis dengan atribut:
- `Open`: harga pembukaan
- `High`: harga tertinggi dalam satu hari
- `Low`: harga terendah dalam satu hari
- `Close`: harga penutupan
- `Volume`: volume transaksi

Dataset terdiri dari 1.211 data observasi.

### Exploratory Data Analysis (EDA)

Untuk memahami karakteristik dasar dari dataset harga saham yang digunakan, dilakukan analisis statistik deskriptif terhadap fitur-fitur numerik.

![Statistik Deskriptif Harga Saham](a70b26b4-f168-4828-9735-cf51e4b3ee28.png)

**Penjelasan Tabel:**

- Dataset terdiri dari **1.211** data entri.
- Rata-rata harga penutupan (`Close`) adalah **1148.68**, dengan standar deviasi **610.16**.
- Harga penutupan minimum: **237.50**, maksimum: **2760.00**
- Volume perdagangan sangat bervariasi, rata-rata **9 juta** dengan maksimum **1.04 miliar**.

## Data Preparation

Langkah-langkah preprocessing yang dilakukan meliputi:
- Menghapus nilai null atau tidak valid
- Normalisasi fitur numerik menggunakan Min-Max Scaling
- Membagi data menjadi data latih dan data uji dengan rasio 80:20

## Modeling

Tiga model machine learning yang digunakan:
- **KNN**: Algoritma sederhana berbasis tetangga terdekat
- **Random Forest**: Model ensambel berbasis pohon keputusan
- **Gradient Boosting**: Model boosting yang menggabungkan banyak pohon lemah secara iteratif

| Model     | RMSE Train | RMSE Test |
|-----------|------------|-----------|
| KNN       | 0.292776   | 0.245496  |
| RF        | 0.049505   | 0.207430  |
| Boosting  | 5.849046   | 5.064820  |

**Analisis:**
- RF memberikan performa terbaik di data uji.
- Boosting menunjukkan overfitting karena error train sangat kecil namun test error besar.

## Evaluation

### Metrik Evaluasi: RMSE

RMSE (Root Mean Squared Error) digunakan karena metrik ini menunjukkan seberapa besar deviasi antara nilai prediksi dan nilai aktual dalam satuan aslinya.

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
\]

- Semakin kecil RMSE, semakin baik model dalam melakukan prediksi.
- Model Random Forest memiliki RMSE test paling rendah, sehingga dipilih sebagai model terbaik untuk prediksi harga saham ini.
