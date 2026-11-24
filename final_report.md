# Laporan Akhir Proyek: Predicting Annual Income from Census Data
**Klasifikasi Biner Tersupervisi Menggunakan Sensus Dewasa Amerika Serikat 1994**

---

## 1. Ringkasan Eksekutif

Proyek ini bertujuan untuk membangun, mengevaluasi, menginterpretasikan, dan mengaudit model pembelajaran mesin (*supervised binary classification*) untuk memprediksi apakah pendapatan tahunan seseorang melebihi USD 50.000 (`>50K`) atau tidak (`<=50K`) berdasarkan data sensus historis Amerika Serikat tahun 1994 (*Adult Census Dataset*, UCI Machine Learning Repository).

Dalam proyek ini, dilakukan investigasi kualitas data, pembersihan data secara terkontrol, pipeline rekayasa fitur tanpa kebocoran data (*data leakage*), validasi silang berstrata 5-lipatan (*5-fold stratified cross-validation*), perbandingan 5 famili algoritma (Dummy Classifier, Logistic Regression, Decision Tree, Random Forest, dan HistGradientBoosting), penanganan ketidakseimbangan kelas (*class imbalance*), penalaan hiperparameter (*GridSearchCV*), evaluasi akhir pada dataset pengujian yang tidak tersentuh (*held-out test set*), interpretasi model (*permutation feature importance*), serta audit keadilan algoritmik (*fairness and demographic disparity audit*) terhadap kelompok `sex` dan `race`.

Model terbaik yang terpilih adalah **Tuned HistGradientBoostingClassifier** yang diintegrasikan dalam pipeline *scikit-learn* bersama *ColumnTransformer* (imputasi median dan *standard scaling* untuk fitur numerik; imputasi modus dan *one-hot encoding* untuk fitur kategorikal). Pada dataset pengujian independen ($N = 9.758$), model ini meraih:
- **Accuracy:** 87,83%
- **ROC-AUC:** 0,9327
- **PR-AUC (Average Precision):** 0,8408
- **F1-Score Kelas Positif (`>50K`):** 0,7293
- **Precision (`>50K`):** 77,97%
- **Recall (`>50K`):** 68,49%
- **Balanced Accuracy:** 81,20%

---

## 2. Identitas Proyek dan Definisi Masalah

- **Judul Proyek:** *Predicting Annual Income from Census Data*
- **Tipe Pembelajaran:** *Supervised Learning*
- **Tugas:** *Binary Classification*
- **Target:** `income`
  - Kelas Negatif (`<=50K`): Pendapatan tahunan $\le$ USD 50.000.
  - Kelas Positif (`>50K`): Pendapatan tahunan $>$ USD 50.000.
- **Asumsi Kausalitas:** Model hanya mempelajari korelasi statistik dan pola asosiasi dalam sampel historis sensus. Tidak ada klaim bahwa variabel prediktor (seperti status pernikahan atau pendidikan) secara kausal menyebabkan seseorang memperoleh pendapatan tertentu.
- **Batasan Penggunaan:** Dataset dikumpulkan pada tahun 1994 oleh Barry Becker dan Ronny Kohavi dari biro sensus AS. Hasil eksperimen ini merupakan demonstrasi akademis dan metodologis, bukan cerminan kondisi ekonomi saat ini. Model **dilarang keras** untuk digunakan dalam pengambilan keputusan penting terhadap individu di dunia nyata, termasuk penerimaan kerja (*hiring*), pemberian kredit perbankan, premi asuransi, atau kelayakan bantuan sosial.

---

## 3. Pemeriksaan Dataset dan Temuan Awal

Dataset yang digunakan bersumber dari UCI Machine Learning Repository dengan rincian awal:
- **Jumlah Observasi Awal:** 48.842 baris
- **Jumlah Kolom:** 15 kolom (14 fitur prediktor, 1 target `income`)
- **Fitur Numerik (6):** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
- **Fitur Kategorikal (8):** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.
- **Target (1):** `income`.

### Pemeriksaan Kualitas Data Awal:
1. **Ketidakkonsistenan Nilai Target:** Terdapat variasi string dengan tanda titik di akhir label (`<=50K` sebanyak 24.720; `<=50K.` sebanyak 12.435; `>50K` sebanyak 7.841; `>50K.` sebanyak 3.846).
2. **Missing Values dan Simbol Tanda Tanya (`?`):**
   - Kolom `workclass`: 963 nilai kosong (*null*) dan 1.836 simbol `?` (total 2.799 nilai tidak diketahui).
   - Kolom `occupation`: 966 nilai kosong (*null*) dan 1.843 simbol `?` (total 2.809 nilai tidak diketahui).
   - Kolom `native-country`: 274 nilai kosong (*null*) dan 583 nilai `?` (total 857 nilai tidak diketahui).
3. **Baris Duplikat Identik:**
   - Sebelum normalisasi string, teridentifikasi 29 baris duplikat.
   - Setelah perapian spasi dan normalisasi label target, teridentifikasi **52 baris duplikat identik** pada seluruh kolom.

---

## 4. Pembersihan Data dan Keputusan Metodologis

Pembersihan data dilakukan dengan prinsip reproduktifitas dan pencegahan kebocoran data (*data leakage*):

1. **Perapian String:** Spasi di awal dan akhir teks (*leading/trailing whitespace*) dipangkas pada seluruh kolom bertipe string.
2. **Standardisasi Nilai Hilang:** Seluruh karakter placeholder `?`, string `nan`, dan `None` dikonversi menjadi `np.nan`. Tidak ada baris yang dihapus semata-mata karena memuat *missing values*; strategi imputasi ditangani sepenuhnya di dalam pipeline.
3. **Normalisasi Target:** Tanda titik di akhir label dihapus. Dilakukan verifikasi ketat sehingga target hanya terdiri dari tepat dua kelas: `<=50K` dan `>50K`. Tidak ditemukan label asing di luar kedua kelas tersebut.
4. **Penanganan Baris Duplikat Identik:** Sebanyak 52 baris duplikat identik dihapus dari dataset sehingga jumlah baris bersih akhir menjadi **48.790 baris**. Alasan penghapusan: baris duplikat identik kemungkinan besar merupakan artefak pencatatan ganda pada survei sensus; mempertahankannya dapat menyebabkan *data leakage* di mana observasi yang identik terbagi ke dalam fold pelatihan dan pengujian.
5. **Distribusi Kelas Akhir Pasca-Pembersihan:**
   - Kelas `<=50K`: 37.109 observasi (76,06%)
   - Kelas `>50K`: 11.681 observasi (23,94%)
   - Total: 48.790 observasi.
6. **Keputusan Fitur `education` vs `education-num`:**
   Pemeriksaan silang membuktikan adanya hubungan bijeksi 1:1 antara kategori teks `education` dan nilai integer `education-num` (Preschool = 1 hingga Doctorate = 16). Memasukkan kedua fitur secara bersamaan akan memberikan bobot ganda (*redundancy/collinearity*). Oleh karena itu, fitur `education` (string) dihapus, dan `education-num` dipertahankan sebagai fitur numerik ordinal yang secara alami merefleksikan jenjang tahun pendidikan formal.
7. **Keputusan Fitur `fnlwgt` (Final Weight):**
   `fnlwgt` adalah variabel pembobotan statistik yang dirancang oleh Biro Sensus AS agar sampel mencerminkan estimasi total populasi pada tingkat geografis dan demografis negara bagian. `fnlwgt` bukan karakteristik individual dan memiliki korelasi yang hampir nol terhadap pendapatan individu ($r = -0,0063$). Memasukkan `fnlwgt` hanya menambahkan *noise* bervariansi tinggi yang tidak dapat digeneralisasi. Oleh karena itu, `fnlwgt` dikeluarkan dari fitur prediktor model.
8. **Nilai Ekstrem (Outlier):**
   Pada `capital-gain`, nilai 99.999 mencerminkan pembatasan batas atas sensus (*top-coding* untuk keuntungan modal di atas $100K). Pada `hours-per-week`, nilai berkisar antara 1 hingga 99 jam. Nilai-nilai ini adalah data sensus yang valid dan tidak dihapus secara sewenang-wenang.

---

## 5. Eksplorasi Data dan Visualisasi Wajib (Fig 1 - Fig 8)

Seluruh 14 grafik wajib telah disimpan di direktori `figures/` dalam format PNG dengan resolusi tinggi (200 DPI). Berikut interpretasi untuk visualisasi EDA (Fig 1 hingga Fig 8):

| No. | File Visualisasi | Temuan Kunci |
|:---:|:---|:---|
| 1 | `01_income_class_distribution.png` | Memperlihatkan ketidakseimbangan kelas target: 76,06% kelas `<=50K` ($n=37.109$) dan 23,94% kelas `>50K` ($n=11.681$). Rasio sekitar 3,17 : 1 menegaskan bahwa metrik akurasi saja dapat menyesatkan (*accuracy paradox*). |
| 2 | `02_missing_values_per_feature.png` | Menunjukkan kelengkapan fitur. Fitur yang memuat nilai hilang / tidak diketahui adalah `occupation` (2.809 baris; 5,76%), `workclass` (2.799 baris; 5,74%), dan `native-country` (857 baris; 1,76%). Fitur lainnya 100% lengkap. |
| 3 | `03_age_distribution_by_income.png` | Distribusi usia kelompok `<=50K` condong ke usia muda dengan median 34 tahun, sedangkan kelompok `>50K` bergeser ke usia matang dengan median 44 tahun. Rentang usia produktif 35–55 tahun memiliki konsentrasi pendapatan tinggi terbesar. |
| 4 | `04_education_level_by_income.png` | Terdapat korelasi positif yang sangat kuat antara jenjang pendidikan dan proporsi pendapatan tinggi: jenjang sekolah dasar ($<9$th grade) memiliki $<6$% `>50K`, sarjana (*Bachelors*) mencapai 41,3%, magister (*Masters*) 55,7%, dan lulusan doktoral (*Doctorate*) serta *Prof-school* melampaui 72%. |
| 5 | `05_occupation_by_income.png` | Kategori profesi manajerial dan spesialis (*Exec-managerial* dan *Prof-specialty*) memiliki proporsi `>50K` tertinggi ($>45$%), sedangkan profesi jasa seperti *Other-service*, *Priv-house-serv*, dan *Handlers-cleaners* memiliki proporsi `>50K` di bawah 10%. |
| 6 | `06_hours_per_week_by_income.png` | Jam kerja standar berkonsentrasi pada 40 jam/minggu untuk kedua kelompok. Namun, individu dengan pendapatan `>50K` memiliki median 40 jam dengan ekor distribusi yang tebal pada rentang 45–60 jam/minggu. Sebaliknya, pekerja paruh waktu ($<35$ jam/minggu) didominasi oleh kelas `<=50K`. |
| 7 | `07_capital_gain_loss_distribution.png` | Distribusi keuntungan dan kerugian modal mengalami inflasi nol (*zero-inflation*) ekstrem: 91,7% data memiliki `capital-gain = 0` dan 95,3% memiliki `capital-loss = 0`. Pada data non-nol, terlihat lonjakan tegas pada nilai *top-coded* $99.999. |
| 8 | `08_numeric_correlation_heatmap.png` | Korelasi linier tertinggi terhadap indikator pendapatan biner (`>50K`) dicatatkan oleh `education-num` ($r = 0,333$), disusul `age` ($r = 0,230$), `hours-per-week` ($r = 0,228$), dan `capital-gain` ($r = 0,223$). Korelasi antarfitur prediktor relatif rendah ($r < 0,15$), menandakan tidak adanya multikolinieritas ekstrem di antara fitur numerik. |

---

## 6. Desain Pipeline dan Pembagian Data

- **Metode Pemisahan:** Data dibagi menjadi data latih (*training set*) sebesar 80% ($N = 39.032$) dan data uji (*test set*) sebesar 20% ($N = 9.758$) dengan stratifikasi target (`stratify=y`) dan `random_state=42`.
- **Pencegahan Data Leakage:** Pembagian data dilakukan secara ketat sebelum proses imputasi, kalkulasi skala, atau pengkodean kategori. Parameter transformasi (`median` untuk numerik, `most_frequent` untuk kategorikal, serta `mean` dan `std` untuk *StandardScaler*) dihitung hanya dari lipatan data latih.
- **Arsitektur Preprocessing (`ColumnTransformer`):**
  - **Jalur Numerik (`DEFAULT_NUMERIC_FEATURES`):** `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`. Ditransformasikan menggunakan `SimpleImputer(strategy='median')` dilanjutkan dengan `StandardScaler()`.
  - **Jalur Kategorikal (`DEFAULT_CATEGORICAL_FEATURES`):** `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`. Ditransformasikan menggunakan `SimpleImputer(strategy='most_frequent')` dilanjutkan dengan `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

---

## 7. Eksperimen Model dan Validasi Silang (Training Set)

Sebanyak 9 konfigurasi model dievaluasi menggunakan *5-fold Stratified Cross-Validation* pada data latih. Hasil out-of-fold cross-validation dirangkum pada tabel berikut (diurutkan berdasarkan PR-AUC):

| Model | Accuracy | Precision | Recall | F1-Score (`>50K`) | Macro F1 | Balanced Acc | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **HistGradientBoosting (Default)** | **0,8706** | **0,7725** | 0,6514 | **0,7068** | **0,8119** | 0,7955 | **0,9271** | **0,8254** |
| **HistGradientBoosting (Balanced)** | 0,8323 | 0,6046 | **0,8661** | 0,7121 | 0,7969 | **0,8439** | 0,9272 | 0,8252 |
| **Random Forest (Default)** | 0,8620 | 0,7901 | 0,5768 | 0,6668 | 0,7899 | 0,7643 | 0,9157 | 0,8019 |
| **Random Forest (Balanced)** | 0,8074 | 0,5634 | 0,8701 | 0,6839 | 0,7727 | 0,8289 | 0,9145 | 0,7924 |
| **Decision Tree (Default)** | 0,8539 | 0,7430 | 0,5956 | 0,6612 | 0,7840 | 0,7654 | 0,9000 | 0,7632 |
| **Logistic Regression (Default)** | 0,8507 | 0,7308 | 0,5957 | 0,6564 | 0,7805 | 0,7633 | 0,9042 | 0,7613 |
| **Logistic Regression (Balanced)** | 0,8067 | 0,5645 | 0,8423 | 0,6760 | 0,7691 | 0,8189 | 0,9042 | 0,7592 |
| **Decision Tree (Balanced)** | 0,8000 | 0,5525 | 0,8660 | 0,6746 | 0,7651 | 0,8226 | 0,8955 | 0,7539 |
| **Dummy Classifier (Baseline)** | 0,7606 | 0,0000 | 0,0000 | 0,0000 | 0,4320 | 0,5000 | 0,5000 | 0,2394 |

### Analisis Dampak Penanganan Ketidakseimbangan Kelas (*Class Imbalance*):
- Penggunaan `class_weight='balanced'` pada seluruh algoritma menghasilkan trade-off yang konsisten dan dapat diprediksi: **Recall melonjak drastis** (misal pada HistGradientBoosting dari 65,14% menjadi 86,61%; pada Logistic Regression dari 59,57% menjadi 84,23%), namun **Precision menurun** dari rentang 73–77% ke 56–60%.
- Nilai PR-AUC dan ROC-AUC tidak mengalami perubahan substansial karena kedua metrik ini mengevaluasi kualitas pemeringkatan probabilitas secara independen dari ambang batas (*threshold* 0,5).
- Model *ensemble boosting* (`HistGradientBoosting`) menunjukkan dominasi nyata atas model linear dan pohon keputusan tunggal pada seluruh metrik komprehensif.

---

## 8. Penalaan Hiperparameter (*Hyperparameter Tuning*)

Penalaan dilakukan pada model terbaik, `HistGradientBoostingClassifier`, menggunakan `GridSearchCV` 5-lipatan pada data latih dengan fungsi skor *Average Precision* (PR-AUC) pada kelas positif `>50K`.

Ruang pencarian dan parameter terbaik yang diperoleh:
- `learning_rate`: [0.08, 0.10, 0.12] $\rightarrow$ **0.08**
- `max_iter`: [100, 150] $\rightarrow$ **150**
- `max_leaf_nodes`: [31, 45] $\rightarrow$ **31**
- `l2_regularization`: [0.0, 1.0] $\rightarrow$ **1.0**

Skor validasi silang PR-AUC meningkat dari 0,8254 menjadi **0,8258**. Model penalaan akhir kemudian dilatih ulang pada seluruh data latih ($N = 39.032$) dan diserialisasi ke `models/best_income_pipeline.joblib`.

---

## 9. Evaluasi Akhir pada Dataset Pengujian Independen (Test Set)

Evaluasi akhir dilakukan **satu kali** terhadap dataset pengujian independen ($N = 9.758$; 7.422 `<=50K` dan 2.336 `>50K`).

### Tabel Perbandingan Kinerja Lengkap pada Data Uji:

| Model | Accuracy | Precision (`>50K`) | Recall (`>50K`) | F1-Score (`>50K`) | Macro F1 | Balanced Acc | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Best Tuned HistGB** | **0,8783** | **0,7797** | 0,6849 | **0,7293** | **0,8254** | 0,8120 | **0,9327** | **0,8408** |
| HistGradientBoosting (Default) | 0,8756 | 0,7780 | 0,6721 | 0,7212 | 0,8206 | 0,8059 | 0,9323 | 0,8396 |
| HistGradientBoosting (Balanced) | 0,8324 | 0,6037 | **0,8737** | 0,7140 | 0,7978 | **0,8466** | 0,9326 | 0,8394 |
| Random Forest (Default) | 0,8688 | 0,8021 | 0,6002 | 0,6866 | 0,8018 | 0,7768 | 0,9212 | 0,8137 |
| Random Forest (Balanced) | 0,8082 | 0,5644 | 0,8707 | 0,6848 | 0,7735 | 0,8296 | 0,9173 | 0,7989 |
| Decision Tree (Default) | 0,8654 | 0,7778 | 0,6130 | 0,6857 | 0,8000 | 0,7790 | 0,9058 | 0,7733 |
| Logistic Regression (Default) | 0,8537 | 0,7326 | 0,6122 | 0,6670 | 0,7866 | 0,7709 | 0,9076 | 0,7683 |
| Logistic Regression (Balanced) | 0,8065 | 0,5633 | 0,8532 | 0,6786 | 0,7701 | 0,8225 | 0,9074 | 0,7660 |
| Decision Tree (Balanced) | 0,7995 | 0,5513 | 0,8741 | 0,6762 | 0,7655 | 0,8251 | 0,9004 | 0,7546 |
| Dummy Classifier (Baseline) | 0,7606 | 0,0000 | 0,0000 | 0,0000 | 0,4320 | 0,5000 | 0,5000 | 0,2394 |

### Analisis Evaluasi Visual (Fig 9 - Fig 12):
- **Fig 9 (`09_model_metrics_comparison.png`):** Memperlihatkan perbandingan metrik multi-dimensi. Model tuned boosting mendominasi semua metrik keseimbangan dan AUC.
- **Fig 10 (`10_best_model_confusion_matrix.png`):** Dari 7.422 sampel aktual `<=50K`, model berhasil memprediksi benar sebanyak 6.970 sampel (93,9% *specificity*) dan hanya salah memprediksi 452 sampel (6,1% *false positive*). Dari 2.336 sampel aktual `>50K`, model berhasil mendeteksi 1.600 sampel (68,5% *true positive/recall*) dan 736 sampel *false negative*.
- **Fig 11 (`11_roc_curves.png`):** Model Tuned HistGB dan default HistGB mencapai ROC-AUC tertinggi (0,933), jauh mengungguli garis diagonal baseline acak (0,500) dan model linear (0,908).
- **Fig 12 (`12_precision_recall_curves.png`):** Mengingat ketidakseimbangan kelas, kurva PR adalah indikator paling andal. Model Tuned HistGB mempertahankan kurva presisi tinggi pada berbagai tingkat recall dengan PR-AUC 0,841, melampaui jauh prevalensi dasar 0,239.

---

## 10. Interpretabilitas Model (*Feature Importance*)

Kontribusi fitur dianalisis menggunakan *Permutation Feature Importance* pada data uji independen dengan mengukur penurunan nilai PR-AUC (10 kali pengulangan):

| Peringkat | Fitur | Mean Permutation Importance (Penurunan PR-AUC) | Standar Deviasi |
|:---:|:---|:---:|:---:|
| 1 | `capital-gain` | **0,2436** | 0,0050 |
| 2 | `marital-status` | **0,1376** | 0,0044 |
| 3 | `education-num` | **0,0676** | 0,0036 |
| 4 | `age` | **0,0631** | 0,0039 |
| 5 | `capital-loss` | **0,0459** | 0,0025 |
| 6 | `occupation` | **0,0307** | 0,0017 |
| 7 | `hours-per-week` | **0,0202** | 0,0027 |
| 8 | `relationship` | **0,0093** | 0,0012 |
| 9 | `workclass` | **0,0069** | 0,0008 |
| 10 | `native-country` | **0,0022** | 0,0006 |
| 11 | `race` | **0,0014** | 0,0006 |
| 12 | `sex` | **0,0013** | 0,0005 |

### Interpretasi (Fig 13 - `13_feature_importance.png`):
- `capital-gain` adalah prediktor individual terkuat. Nilai keuntungan modal yang substansial hampir selalu berasosiasi eksklusif dengan kelas pendapatan tinggi.
- `marital-status` (terutama kategori `Married-civ-spouse`) menempati posisi kedua. Dalam data sensus 1994, kepala keluarga menikah memiliki tingkat stabilitas finansial dan akumulasi aset yang jauh lebih tinggi.
- `education-num` dan `age` menegaskan peran akumulasi modal manusia (*human capital*) dan pengalaman kerja seiring berjalannya usia.
- Fitur sensitif seperti `sex` dan `race` memiliki kepentingan permutasi langsung yang sangat kecil ($<0,0015$). Namun, sebagaimana diuraikan pada audit fairness di bawah, hal ini tidak berarti model bebas dari pengaruh demografis.

---

## 11. Pemeriksaan Keadilan dan Disparitas Demografis (*Fairness Audit*)

Untuk memeriksa potensi bias algoritmik, dilakukan audit performa antarsubgrup pada data uji ($N = 9.758$) dan membandingkan **Model Lengkap** dengan **Model Ablasi** (model yang dilatih tanpa fitur `sex` dan `race`):

### Tabel Hasil Audit Keadilan Antarsubgrup (Test Set):

| Varian Model | Tipe Grup | Subgrup | Ukuran Sampel ($N$) | Selection Rate | Precision | Recall (TPR) | FPR | F1-Score | Accuracy |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model Lengkap** | **Sex** | **Female** | 3.326 | 7,88% | 83,59% | 57,18% | 1,46% | 0,6791 | 93,78% |
| Model Ablasi | Sex | Female | 3.326 | 7,97% | 83,40% | 57,70% | 1,50% | 0,6821 | 93,81% |
| **Model Lengkap** | **Sex** | **Male** | 6.432 | 27,83% | 77,15% | 70,71% | 9,13% | 0,7379 | 84,75% |
| Model Ablasi | Sex | Male | 6.432 | 27,55% | 77,20% | 70,05% | 9,02% | 0,7345 | 84,62% |
| **Model Lengkap** | **Race** | **White** | 8.353 | 22,54% | 78,28% | 69,23% | 6,57% | 0,7348 | 87,26% |
| Model Ablasi | Race | White | 8.353 | 22,27% | 78,39% | 68,48% | 6,46% | 0,7310 | 87,15% |
| **Model Lengkap** | **Race** | **Black** | 916 | 8,19% | 78,67% | 52,21% | 1,99% | 0,6277 | 92,36% |
| Model Ablasi | Race | Black | 916 | 9,17% | 76,19% | 56,64% | 2,49% | 0,6497 | 92,47% |
| **Model Lengkap** | **Race** | **Asian-Pac-Islander** | 311 | 24,76% | 67,53% | 73,24% | 10,42% | 0,7027 | 85,85% |
| Model Ablasi | Race | Asian-Pac-Islander | 311 | 24,12% | 69,33% | 73,24% | 9,58% | 0,7123 | 86,50% |
| **Model Lengkap** | **Race** | **Amer-Indian-Eskimo** | 94 | 10,64% | 90,00% | 69,23% | 1,23% | 0,7826 | 94,68% |
| Model Ablasi | Race | Amer-Indian-Eskimo | 94 | 11,70% | 81,82% | 69,23% | 2,47% | 0,7500 | 93,62% |
| **Model Lengkap** | **Race** | **Other** | 84 | 8,33% | 85,71% | 60,00% | 1,35% | 0,7059 | 94,05% |
| Model Ablasi | Race | Other | 84 | 8,33% | 85,71% | 60,00% | 1,35% | 0,7059 | 94,05% |

### Temuan Kritis Fairness (Fig 14 - `14_fairness_by_sex_and_race.png`):
1. **Disparitas Demografis Historis:**
   - Tingkat seleksi (*selection rate*) pada laki-laki (27,83%) jauh lebih tinggi dibandingkan perempuan (7,88%). Hal ini merefleksikan kesenjangan upah dan partisipasi angkatan kerja dalam data sensus 1994, di mana proporsi `>50K` aktual pada perempuan hanya ~11% berbanding ~31% pada laki-laki.
   - Pada subgrup ras, kelompok *Black* memiliki tingkat seleksi 8,19% dan recall 52,21%, jauh lebih rendah dibandingkan kelompok *White* (seleksi 22,54%, recall 69,23%) dan *Asian-Pac-Islander* (seleksi 24,76%, recall 73,24%).
2. **Efek Variabel Proxy (*The Proxy Variable Problem*):**
   - Ketika atribut `sex` dan `race` dihapus secara total dari pelatihan model (*Ablation Model*), **disparitas metrik antarkelompok hampir tidak berubah**: selection rate perempuan hanya bergeser dari 7,88% ke 7,97%, dan laki-laki dari 27,83% ke 27,55%.
   - Fenomena ini membuktikan secara empiris bahwa menghapus variabel sensitif (*fairness through unawareness*) **tidak efektif** menghilangkan bias algoritmik. Fitur lain seperti `marital-status`, `relationship` (misalnya label *Husband* vs *Wife*), `occupation`, dan `hours-per-week` bertindak sebagai *proxy* kuat yang tetap merekonstruksi informasi gender dan latar belakang demografis.
3. **Ketidakpastian Ukuran Sampel Kecil:**
   - Subgrup seperti `Amer-Indian-Eskimo` ($N=94$) dan `Other` ($N=84$) memiliki ukuran sampel yang sangat terbatas dalam test set. Fluktuasi metrik pada kelompok ini harus diinterpretasikan secara hati-hati karena memiliki margin kesalahan statistik (*sampling error*) yang tinggi.

---

## 12. Kesimpulan dan Rekomendasi

1. **Keberhasilan Metodologis:** Pipeline klasifikasi biner berhasil dibangun secara end-to-end tanpa kebocoran data (*zero data leakage*), dengan verifikasi silang berulang, penalaan terarah, dan pencapaian performa tinggi (Akurasi 87,83%, ROC-AUC 0,9327, PR-AUC 0,8408).
2. **Keterbatasan Utama:**
   - **Temporalitas:** Dataset sensus 1994 sudah berusia lebih dari tiga dekade. Nilai nominal USD 50.000 pada tahun 1994 setara dengan lebih dari USD 100.000 pada nilai mata uang saat ini akibat inflasi kumulatif.
   - **Inflasi Nol dan Top-Coding:** Keuntungan modal (*capital-gain*) memiliki 91,7% nilai nol dan dibatasi pada $99.999, membatasi resolusi prediksi pada spektrum kekayaan ekstrem.
   - **Disparitas Sistemik:** Model mereproduksi ketimpangan struktural sosial-ekonomi yang terekam dalam data historis.
3. **Pernyataan Penutup:** Model ini berfungsi dengan sangat baik sebagai tolok ukur akademis dan demonstrasi kemampuan rekayasa pipeline machine learning modern. Namun, model ini **tidak boleh diterapkan dalam sistem otomasi penentuan nasib manusia** di dunia nyata.
