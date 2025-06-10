
# Laporan Prediksi Harga Rumah Menggunakan Machine Learning - Vittorio Fiorentino

---

# Domain Proyek

Pergerakan harga properti atau rumah merupakan aspek penting dalam dunia real estat, baik bagi pembeli, penjual, maupun investor. Dalam era digital dan data-driven saat ini, prediksi harga rumah berbasis data historis menjadi pendekatan yang sangat diminati karena mampu memberikan wawasan yang lebih akurat dalam pengambilan keputusan jual beli atau investasi.

Dengan meningkatnya ketersediaan data properti (seperti ukuran rumah, lokasi, jumlah kamar, dan tahun pembangunan), serta kemajuan teknologi machine learning, analisis prediktif harga rumah menjadi solusi yang efektif dan efisien dalam memahami tren pasar properti.


---

# Business Understanding

## Problem Statements

1. Bagaimana memprediksi harga rumah berdasarkan data historis dan karakteristik rumah?

2. Model machine learning mana yang memberikan performa terbaik dalam memprediksi harga rumah?

## Goals

1. Menghasilkan model prediktif yang mampu memperkirakan harga rumah berdasarkan fitur-fitur seperti luas bangunan, jumlah kamar, lokasi, kondisi, dan lainnya.

2. Membandingkan performa beberapa algoritma machine learning untuk menentukan model terbaik dalam memprediksi harga rumah.

## Solution Statement

Untuk mencapai tujuan di atas, digunakan tiga algoritma machine learning, yaitu:
- K-Nearest Neighbors (KNN)

Digunakan untuk memprediksi harga rumah berdasarkan kedekatan fitur rumah lain dengan harga yang telah diketahui.

- Random Forest (RF)

Model ensemble berbasis pohon keputusan yang dapat menangani data kompleks dan memberikan hasil prediksi yang stabil.

- Gradient Boosting

Model boosting yang secara bertahap memperbaiki kesalahan prediksi dari model sebelumnya untuk meningkatkan akurasi.

Evaluasi model dilakukan menggunakan Root Mean Squared Error (RMSE) pada data pelatihan dan pengujian.


---

# Data Understanding

Dataset diperoleh dari kaggle 'https://www.kaggle.com/datasets/shivachandel/kc-house-data'

Dataset yang digunakan merupakan data rumah/properti historis dengan atribut:
- Jumlah baris = 21613 baris
- Jumlah kolom = 21 kolom
- Terdapat missing value pada variabel sqft_above
- Semua kolom bertipe Numeric (Kolom Date harusnya bertipe datetime)

**Insight**

- Kita akan berfokus pada beberapa kolom variabel saja, sehingga variable yang tidak perlu akan di drop.



---


## Exploratory Data Analysis (EDA)

Untuk memahami karakteristik dasar dari dataset rumah yang digunakan, dilakukan analisis statistik deskriptif terhadap fitur-fitur numerik.

Penjelasan 21 Fitur Awal:
1. id
- ID unik untuk setiap properti. Digunakan sebagai identifikasi individual dalam dataset.

2. date
- Tanggal penjualan rumah. Format umum: YYYYMMDDT000000.

3. price
- Harga jual rumah (dalam USD). Ini adalah variabel target jika ingin melakukan prediksi harga.

4. bedrooms
- Jumlah kamar tidur di rumah.

5. bathrooms
- Jumlah kamar mandi (dinyatakan dalam angka desimal; misal 1.5 berarti 1 kamar mandi penuh dan 1 kamar mandi kecil).

6. sqft_living
- Luas area yang dapat dihuni (livable) dalam satuan kaki persegi.

7. sqft_lot
- Luas total tanah properti (termasuk bangunan dan halaman), dalam kaki persegi.

8. floors
- Jumlah lantai bangunan rumah.

9. waterfront
- Indikator apakah rumah memiliki pemandangan langsung ke air (danau, laut, dll).
-  Nilai: 0 (tidak), 1 (ya).

10. view
- Indeks visualisasi rumah terhadap pemandangan luar. Skala 0–4; semakin tinggi, semakin bagus pandangan rumah.

11. condition
- Kondisi umum rumah (bukan renovasi). Skala 1–5; 1 paling buruk, 5 sangat baik.

12. grade
- Penilaian kualitas konstruksi dan desain rumah oleh King County (bukan kondisi). Skala 1–13.

13. sqft_above
- Luas area bangunan di atas tanah (tidak termasuk basement), dalam kaki persegi.

14. sqft_basement
- Luas area basement (bawah tanah), dalam kaki persegi.

15. yr_built
- Tahun rumah pertama kali dibangun.

16. yr_renovated
- Tahun terakhir rumah direnovasi. Jika tidak pernah direnovasi, bernilai 0.

17. zipcode
- Kode pos wilayah rumah berada.

18. lat
- Latitude (garis lintang) lokasi properti.

19. long
- Longitude (garis bujur) lokasi properti.

20. sqft_living15
- Rata-rata luas area livable dari 15 rumah tetangga terdekat.

21. sqft_lot15
- Rata-rata luas tanah dari 15 rumah tetangga terdekat.


**Struktur Data**

- Terdapat missing Value pada sqft_above dengan jumlah 21611 dari 21613
- Semua kolom bertipe Numeric (Kolom Date harusnya bertipe datetime)

### Handling Missing Value

Code dibawah untuk menujukkan missing value pada semua kolom.

house.isna().sum()

Output missing value: | sqft_above   |   2  |

### Handling Outliers

**Plot Sebelum di Handling**

![outlier](https://github.com/user-attachments/assets/4ed4aae8-48b9-4d1b-82a1-56fddfc5faf2)

Penjelasan BoxPlot:

- bathrooms Boxplot: Terlihat banyak outlier di atas (hingga 8 kamar mandi). Rumah dengan lebih dari 4–5 kamar mandi itu jarang dan bisa dikategorikan outlier.

- sqft_living Boxplot: Outlier ekstrem sangat banyak di sisi atas (hingga >13.000 sqft). Rumah dengan luas >8000 sqft adalah mansion dan sangat tidak umum, bisa mendistorsi model.

- floors Boxplot: Hampir tidak ada outlier. Rentang antara 1 hingga 3.5 lantai, distribusinya cukup stabil.

- bedrooms Boxplot: Outlier bisa terjadi untuk rumah dengan 33 kamar tidur.

- price Boxplot: Harga memang ada outliers. Mungkin karena ada rumah mahal/mansion

**Insight**

Kita akan berfokus pada beberapa kolom variabel saja, sehingga variable yang tidak perlu akan di drop.

- Memperkecil scope variabel pada dataset untuk menyederhanakan model dan menghindari overfitting.
- Menangani Outlier
- Menangani Missing Value

Terdapat dua baris missing value pada kolom sqft_above.

sqft_above  |  2  |

Code dibawah untuk drop baris pada kolom sqft_above yang memiliki missing value.

house = house.dropna(subset=['sqft_above'])

Missing value di drop karena hanya dua baris data dari 21613, sehingga tidak akan berpengaruh terhadap dataset.

---

# Data Preparation

### Drop Missing Value pada sqft_above

Code dibawah untuk drop missing value pada sqft_above

house = house.dropna(subset=['sqft_above'])
     

### Tujuan memperkecil scope variabel:

- Hal ini bertujuan untuk menyederhanakan model dan menghindari overfitting.
- Beberapa variabel yang dirasa tidak terlalu berpengaruh dengan signifikan akan di drop. Sehingga kita hanya mengambil variabel ( price, bedrooms, bathrooms, sqft_living, floors, waterfront, condition, grade, yr_built, yr_renovated)

**Insight:**

- Kolom bedrooms dan bathrooms memiliki data bernilai 0. Sedikit ambigu jika rumah tidak punya bedrooms dan bathrooms. Tapi bisa saja memang ada. Kita berspekulasi bahwa nilai 0 itu tidak mungkin, sehingga akan kita buang.

### Handling Nilai 0 Pada Bedroom dan Bathrooms

- Bedrooms yang bernilai 0 memiliki 13 baris. Dari hasil ini bisa dilihat memang anomali karena grade rumah termasuk mayoritas tinggi. Sehingga data ini akan dibuang/drop.
- Bathrooms yang bernilai 0 memiliki 3 baris. Dari hasil ini bisa dilihat meskipun bathrooms, tetapi masih ada bedrooms. Jika dilihat dari yr_built kisaran 1948-1966 yang dimana termasuk sudah tua dengan grade yang normal. Sehingga masih termasuk normal dan tetap digunakan.


### Handling Outliers dengan Winsorize

Winsorize adalah teknik statistik yang digunakan untuk mengurangi pengaruh outlier ekstrem dalam data dengan cara mengubah (bukan menghapus) nilai-nilai ekstrem tersebut menjadi nilai yang lebih dekat ke nilai tengah (biasanya batas persentil tertentu).

**Plot Setelah di Handling**

![after-out](https://github.com/user-attachments/assets/aa5af3a3-f0c4-4cfa-a1a2-c7427bb32678)

Setelah berhasil menangani outlier dengan winsorize, data sudah stabil. Variabel price merupakan nilai target jadi tetap dibiarkan meskipun ada outlier karena akan mempengaruhi akurasi model. Model justru perlu belajar bahwa properti tertentu memang bisa mahal.

**Agregasi rata-rata harga berdasarkan kategori luas rumah**

![luas rumah](https://github.com/user-attachments/assets/0062aafd-a56c-4184-84e3-e3e659cea0fe)

Berdasarkan hasil visualisasi, berikut adalah beberapa insight yang bisa didapatkan:

Distribusi Harga Jual Berdasarkan Luas Rumah: Bar chart menunjukkan harga jual ('selling_price') untuk berbagai kategori luas rumah ('sqft_living'). Sumbu X adalah luas rumah dan sumbu Y adalah harga jual.

Kategori dengan Harga Jual Terendah: Range 0-1000

Kategori dengan Harga Jual Tertinggi: Range 4001+

Tren Peningkatan Harga Jual: Terlihat jelas ada tren peningkatan harga jual dari kiri ke kanan.

---

### Univariate Analysis

- Analisis Satu Variabel
- Tidak mempelajari hubungan dengan variabel lain

numerical_features = ['price', 'bathrooms', 'bedrooms', 'floors', 'sqft_living', 'yr_built', 'yr_renovated']
categorical_features = ['waterfront', 'condition', 'grade']

#### Categorical Features

Distribusi untuk fitur: waterfront
            
waterfront   jumlah sampel  persentase                        
0                   21437        99.2
1                     163         0.8


Distribusi untuk fitur: condition
           
condition   jumlah sampel  persentase                        
3                  14021        64.9
4                   5678        26.3
5                   1701         7.9
2                    171         0.8
1                     29         0.1


Distribusi untuk fitur: grade
       
grade   jumlah sampel  persentase                        
7               8975        41.6
8               6065        28.1
9               2615        12.1
6               2038         9.4
10              1134         5.2
11               399         1.8
5                242         1.1
12                89         0.4
4                 27         0.1
13                13         0.1
3                  3         0.0


#### Numerical Features

**Histogram**

![uni-num](https://github.com/user-attachments/assets/4c7c5c77-3fdf-42d2-b883-f76fc169cab4)

### Multivariate Analysis

- Analisis Banyak Variabel
- Digunakan untuk menemukan hubungan, korelasi, atau pengaruh antar variabel.

#### Categorical Features

![multi cat1](https://github.com/user-attachments/assets/7db736be-8ded-4394-94a4-bf292acf9a72)

![multi cat2](https://github.com/user-attachments/assets/59ba43ac-d3c4-4c64-a54f-9521a9ac3469)

![multi cat3](https://github.com/user-attachments/assets/8a9cd359-de8f-47bb-aec1-f2b001a07845)


**Penjelasan**:

- Grafik menunjukkan bahwa rumah yang memiliki waterfront (dekat/punya akses ke air, ditandai dengan 1) memiliki rata-rata harga jauh lebih tinggi dibandingkan rumah yang tidak memiliki waterfront (ditandai dengan 0). Artinya, faktor lokasi terhadap air berpengaruh signifikan terhadap harga properti.

- Semakin tinggi nilai condition, rata-rata harga ikut naik:

  - Condition 1–2: sekitar 330 ribu.

  - Condition 3–4: melonjak ke kisaran 520–540 ribu.

  - Condition 5: tertinggi, ±610 ribu.

    Jadi, kondisi yang lebih baik berasosiasi dengan harga rata-rata yang lebih tinggi, dengan lonjakan paling besar terjadi setelah kategori 2.

- Grafik menunjukkan bahwa semakin tinggi grade, rata-rata harga (price) meningkat tajam:

  - Grade 3–8: harga relatif stabil di bawah 700 ribu.

  - Grade 9–11: mulai naik signifikan, tembus lebih dari 1 juta.

  - Grade 12–13: lonjakan tajam, hingga lebih dari 3 juta di grade 13.

    Kesimpulan: grade berpengaruh kuat terhadap harga, terutama di grade tinggi.


#### Numerical Features

**Matrix Correlation**

![multi matrix](https://github.com/user-attachments/assets/3b456eac-014e-4467-bc4e-953e267733dc)

**Penjelasan Matrix Correlation**:

1.  Korelasi terhadap price:
- Tinggi:

  - sqft_living (0.65): Semakin besar luas rumah, semakin tinggi harganya.

  - bathrooms (0.48): Lebih banyak kamar mandi cenderung menaikkan harga.

- Sedang:

  - bedrooms (0.32), floors (0.26)

- Rendah:

  - yr_renovated (0.13)

  - yr_built (0.05): Tahun dibangun hampir tidak berpengaruh langsung terhadap harga.

2. Korelasi antarfungsi lain:
   
- bathrooms dan sqft_living punya korelasi tinggi (0.75), menunjukkan bahwa rumah lebih besar cenderung memiliki lebih banyak kamar mandi.

- yr_built dan yr_renovated berkorelasi negatif (-0.22), artinya rumah yang lebih lama dibangun cenderung lebih sering direnovasi.

Kesimpulan:

Fitur paling relevan terhadap price adalah sqft_living, lalu bathrooms. Fitur seperti yr_built atau yr_renovated punya pengaruh sangat kecil terhadap harga berdasarkan korelasi linier.


---

## Train Test Split

- Train Test Split: Membagi data menjadi data latih dan data uji dengan rasio 90:10
Total of sample in whole dataset: 21600

Total of sample in train dataset: 19440

Total of sample in test dataset: 2160

- Standarisasi Numeric

	bedrooms	bathrooms	sqft_living	floors		yr_built	yr_renovated
2053	-0.430140	-1.526276	-0.930050	0.008579	-0.750244	-0.210792
13754	-0.430140	-1.526276	-1.335406	-0.917881	-0.409601	-0.210792
10970	0.742685	0.551480	-0.047806	0.935040	-0.954630	-0.210792
15890	1.915510	1.590357	2.591028	0.935040	0.305750	-0.210792
14117	0.742685	0.551480	0.441005	0.935040	0.748587	-0.210792


---

# Modeling 

Tiga model machine learning yang digunakan:
- **KNN**: Algoritma sederhana berbasis tetangga terdekat
- **Random Forest**: Model ensambel berbasis pohon keputusan
- **Gradient Boosting**: Model boosting yang menggabungkan banyak pohon lemah secara iteratif

## KNN

Lazy Learning: Tidak melakukan proses pelatihan dalam arti tradisional. Hanya menyimpan data training.

Prediksi:

1. Untuk setiap titik data baru (X_test), hitung jarak (biasanya Euclidean) ke semua data training.

2. Pilih k tetangga terdekat (dalam contohmu, k=7).

3. Ambil rata-rata dari nilai target (y_train) dari 7 tetangga tersebut.


## Random Forest

Cara kerja:

1. Buat banyak pohon keputusan (di sini 50 pohon).

2. Setiap pohon:

  - Dibuat dari sampel acak dengan pengembalian dari data training (bootstrap sampling).

  - Saat membelah node, hanya subset acak dari fitur yang dipertimbangkan.

3. Setiap pohon membuat prediksi, lalu rata-ratanya diambil (untuk regresi).

Parameter penting:

- n_estimators=50: jumlah pohon dalam hutan.

- max_depth=16: membatasi kedalaman pohon → mencegah overfitting.

- n_jobs=-1: gunakan semua core CPU untuk training (paralel).

## Metode Adaptive Boosting

Cara kerja (regresi):

1. Model pertama (weak learner) dilatih pada data asli.

2. Residual (kesalahan prediksi) dihitung.

3. Model selanjutnya dilatih untuk memprediksi residual tersebut, bukan nilai aslinya.

4. Proses berulang: setiap model mencoba memperbaiki kesalahan model sebelumnya.

5. Prediksi akhir adalah gabungan (weighted sum) dari semua model.

Spesifik pada kode:

1. AdaBoostRegressor menggunakan DecisionTreeRegressor depth=1 sebagai default weak learner.

2. learning_rate=0.05: mengatur kontribusi tiap model terhadap prediksi akhir.

3. random_state=55: memastikan hasil yang reprodusibel.


## Scaling Pada Data Uji
Fungsi scaling (normalisasi atau standarisasi) pada data uji (test set) adalah:

Menjadikan skala data uji konsisten dengan data latih, agar model dapat membuat prediksi yang akurat.

**Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1**

X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

---

# Evaluation

## Metrik Evaluasi

			train			test
KNN		| 35029023.438781	| 44108533.120292
RF		| 13031555.816459	| 40826373.091298
Boosting	| 51477538.064462	| 52359369.499679


Penjelasan:

- Boosting memiliki generalisasi terbaik (perbedaan train-test kecil).

- Random Forest akurat di train, tapi test sedikit lebih buruk.

- KNN kurang cocok karena overfitting parah (train bagus, test sangat buruk).

Plot Model

![evaluasi](https://github.com/user-attachments/assets/eb3523a8-dd61-40d0-aaa0-c758a137b6fc)

📌 Interpretasi Singkat:

KNN: MSE di data test cukup tinggi, menunjukkan model sedikit overfitting.

RF: MSE Train rendah sedangkan MSE Test sangat tinggi, menunjukkan model sangat overfitting.

Boosting: MSE Train dan Test hampir sama, menunjukkan model yang paling stabil

✅ Kesimpulan:

KNN dan Boosting lebih cocok digunakan dibanding RF untuk data ini.

Boosting mungkin pilihan terbaik karena Model paling stabil.


## Hasil Pengujian Data

	y_true		prediksi_KNN	prediksi_RF	prediksi_Boosting
21585	270000.0	307818.9	320719.8	383666.0



**Evaluasi Akurasi Prediksi:**

Hasil Asli = 270000
- KNN	307818.9 		 ->	Paling dekat
- RandomForest	320719.8	 ->	Cukup dekat
- Boosting	383666.0	 ->	Cukup jauh dari aslinya

---

## Kesimpulan:

**1. Dampak Model Terhadap Business Understanding**
   
Ketiga model yang diuji: KNN, Random Forest, dan Gradient Boosting—memberikan gambaran yang jelas mengenai bagaimana algoritma machine learning dapat membantu dalam memperkirakan harga rumah secara lebih objektif dan berbasis data historis.

**2. Apakah Problem Statement Terjawab?**

- Bagaimana memprediksi harga rumah berdasarkan data historis dan karakteristik rumah?
  
✔️ Terjawab. Data historis seperti sqft_above, jumlah kamar, dan kondisi rumah berhasil digunakan sebagai fitur dalam model prediksi.

- Bagaimana memprediksi harga rumah berdasarkan data historis dan karakteristik rumah?
  
✔️ Terjawab. Data historis seperti sqft_above, jumlah kamar, dan kondisi rumah berhasil digunakan sebagai fitur dalam model prediksi.

**3. Apakah Goals Tercapai?**

Goal 1:

✔️ Tercapai. Model prediktif telah dibangun berdasarkan fitur-fitur rumah, dengan evaluasi RMSE pada data pelatihan dan pengujian.

Goal 2:

✔️ Tercapai. Performa ketiga model dibandingkan secara objektif. Boosting dipilih sebagai model terbaik karena kemampuan generalisasi-nya paling tinggi, meskipun RMSE-nya tidak paling rendah pada data pelatihan.

