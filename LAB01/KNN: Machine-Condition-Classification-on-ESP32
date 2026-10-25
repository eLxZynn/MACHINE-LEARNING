# Mission-06C — KNN: Machine Condition Classification on ESP32

> **Mata kuliah:** Machine Learning · **Platform:** ESP32 + MicroPython · **Jenis:** klasifikasi biner · **Perangkat tambahan:** tidak ada  
> **Estimasi waktu:** 80–100 menit · **Prasyarat:** konsep train/test split, KNN, normalisasi, accuracy, confusion matrix, menjalankan MicroPython di Thonny.

## 1. Tujuan pembelajaran

Setelah praktikum, mahasiswa mampu:

1. Menjelaskan mengapa KNN dapat berjalan di ESP32 tanpa Scikit-learn dan tanpa fase pelatihan parameter eksplisit.
2. Mengimplementasikan perhitungan jarak, pemilihan **K tetangga terdekat**, dan *majority voting* dengan MicroPython.
3. Menguji pengaruh nilai **K** dan skala fitur pada hasil klasifikasi.
4. Mengevaluasi model dengan **test set terpisah**, accuracy, dan confusion matrix.
5. Mengukur perkiraan *inference latency* dan mengidentifikasi keterbatasan memori perangkat.

## 2. Studi kasus

Bayangkan sebuah mesin industri dipantau berdasarkan dua fitur:

- **Temperature** ($x_1$): temperatur mesin dalam derajat Celsius (°C).
- **Vibration** ($x_2$): tingkat getaran mesin dalam mm/s.

Target klasifikasi:

- **0 = NORMAL**
- **1 = WARNING**

**Penting:** Semua data adalah **sintetik** untuk pembelajaran. Tidak ada sensor yang dihubungkan ke ESP32. Label bukan standar diagnosis kondisi mesin nyata; jangan gunakan model ini untuk keselamatan mesin.

**Alur kerja:** `Data sintetik → penyimpanan di ESP32 → scaling → KNN → prediksi → evaluasi`.

## 3. Alat dan persiapan

- ESP32 yang telah di-*flash* dengan firmware MicroPython.
- Kabel USB data dan komputer dengan Thonny IDE.
- Tidak perlu breadboard, sensor, LED eksternal, koneksi internet, atau paket ML.

Di Thonny, pilih interpreter **MicroPython (ESP32)** dan port serial ESP32. Buat file `main.py` **di perangkat ESP32**. Program akan mencetak hasil di Shell Thonny. Jika sudah ada `main.py` dari praktikum blink, salin/ubah namanya terlebih dahulu agar tidak berjalan bersamaan.

## 4. Ringkasan teori

KNN melakukan klasifikasi dengan mencari **K contoh training yang paling dekat** dengan input baru dan memilih label yang paling banyak muncul di antara tetangga tersebut.

Dengan dua fitur, jarak Euclidean adalah:

$$
d = \sqrt{(x_1-a_1)^2 + (x_2-a_2)^2}
$$

Untuk pengurutan tetangga, akar tidak diperlukan sehingga kita menghitung **jarak kuadrat**:

$$
d^2 = (x_1-a_1)^2 + (x_2-a_2)^2
$$

Keterangan:

- $x_1$, $x_2$: fitur input yang akan diklasifikasikan.
- $a_1$, $a_2$: fitur satu sampel di data training.
- $d$: jarak Euclidean antara kedua sampel.
- $K$: jumlah tetangga yang ikut *voting*.

Karena temperatur dan getaran memiliki rentang angka berbeda, kita gunakan **min–max scaling berdasarkan data training**:

$$
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

> Gunakan rentang minimum dan maksimum **hanya dari training set**. Data test dan input baru menggunakan parameter scaling yang sama. Jangan menyesuaikan scaler menggunakan test set. Nilai data baru yang berada di luar rentang training dapat menghasilkan nilai scaling di luar $[0,1]$; hal ini tidak otomatis salah.

**Catatan konsep:** KNN menyimpan sampel referensi saat disebut “training”. Beban utamanya terjadi ketika **inference**, karena harus menghitung jarak ke sampel referensi.

## 5. Dataset

Modul menyediakan **36 sampel training** dan **12 sampel test** dalam kode. Label dibuat dengan aturan simulasi yang memiliki area batas kelas sehingga prediksi tidak selalu 100% benar. Data test **tidak** digunakan sebagai tetangga saat inference.

Format setiap baris:

```python
(temperature, vibration, label)
```

Contoh: `(68.0, 6.1, 1)` berarti temperatur 68°C, getaran 6,1 mm/s, label WARNING.

## 6. Program utama MicroPython

Simpan kode ini sebagai `main.py` pada ESP32, lalu klik **Run**.

```python
# Mission-06D - KNN pada ESP32 (MicroPython)
import time
import gc

# temperature (C), vibration (mm/s), label (0=NORMAL; 1=WARNING)
TRAIN = [
    (29.0, 1.0, 0),
    (42.4, 2.7, 0),
    (55.8, 4.4, 0),
    (68.0, 6.1, 1),
    (31.4, 7.8, 0),
    (44.8, 1.2, 0),
    (57.0, 2.9, 0),
    (70.4, 4.6, 1),
    (33.8, 6.3, 0),
    (46.0, 8.0, 1),
    (59.4, 1.4, 0),
    (72.8, 3.1, 1),
    (35.0, 4.8, 0),
    (48.4, 6.5, 1),
    (61.8, 8.2, 1),
    (74.0, 1.6, 0),
    (37.4, 3.3, 0),
    (50.8, 5.0, 0),
    (63.0, 6.7, 1),
    (76.4, 8.4, 1),
    (39.8, 1.8, 0),
    (52.0, 3.5, 0),
    (65.4, 5.2, 1),
    (78.8, 6.9, 1),
    (41.0, 8.6, 1),
    (54.4, 2.0, 0),
    (67.8, 3.7, 1),
    (30.0, 5.4, 0),
    (43.4, 7.1, 0),
    (56.8, 8.8, 1),
    (69.0, 2.2, 0),
    (32.4, 3.9, 0),
    (45.8, 5.6, 0),
    (58.0, 7.3, 1),
    (71.4, 9.0, 1),
    (34.8, 2.4, 0),
]
TEST = [
    (32, 1.6, 0),
    (39, 3.1, 0),
    (52, 3.0, 0),
    (60, 4.5, 1),
    (47, 6.5, 1),
    (70, 7.5, 1),
    (78, 8.0, 1),
    (35, 7.9, 0),
    (73, 2.5, 1),
    (57, 5.5, 1),
    (46, 4.4, 0),
    (65, 6.2, 1),
]
LABELS = ("NORMAL", "WARNING")

# Parameter scaling dihitung HANYA dari data training.
MIN_T = min(r[0] for r in TRAIN)
MAX_T = max(r[0] for r in TRAIN)
MIN_V = min(r[1] for r in TRAIN)
MAX_V = max(r[1] for r in TRAIN)
RANGE_T = MAX_T - MIN_T
RANGE_V = MAX_V - MIN_V


def predict_knn(temp, vibration, k=3, scaled=True):
    if not isinstance(k, int) or k < 1 or k > len(TRAIN) or k % 2 == 0:
        raise ValueError("Gunakan K ganjil: 1 sampai jumlah data training")
    distances = []
    for idx, (t, v, label) in enumerate(TRAIN):
        dt = temp - t
        dv = vibration - v
        if scaled:
            dt = dt / RANGE_T
            dv = dv / RANGE_V
        distance_sq = dt * dt + dv * dv
        distances.append((distance_sq, idx, label))

    # Tie jarak dipecahkan secara deterministik memakai urutan data.
    distances.sort(key=lambda r: (r[0], r[1]))
    neighbors = distances[:k]
    warning_votes = sum(r[2] for r in neighbors)
    prediction = 1 if warning_votes > k // 2 else 0
    return prediction, neighbors


def evaluate(k=3, scaled=True, verbose=False):
    correct = 0
    tp = tn = fp = fn = 0
    for temp, vib, actual in TEST:
        predicted, _ = predict_knn(temp, vib, k, scaled)
        correct += predicted == actual
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        else:
            fn += 1
        if verbose:
            print("T=%.1f, V=%.1f | actual=%s predicted=%s" % (
                temp, vib, LABELS[actual], LABELS[predicted]))
    accuracy = 100 * correct / len(TEST)
    print("K=%d, scaled=%s | accuracy=%.2f%% (%d/%d) | TP=%d TN=%d FP=%d FN=%d" % (
        k, scaled, accuracy, correct, len(TEST), tp, tn, fp, fn))
    return accuracy


def demo():
    print("TRAIN =", len(TRAIN), "TEST =", len(TEST))
    print("Rentang training: T=%.1f..%.1f; V=%.1f..%.1f" % (
        MIN_T, MAX_T, MIN_V, MAX_V))
    for k in (1, 3, 5, 7):
        evaluate(k)
    print("\nPengaruh scaling untuk K=3")
    evaluate(3, scaled=False)
    evaluate(3, scaled=True)


def interactive():
    print("Ketik q untuk keluar. Input sintetik, BUKAN pembacaan sensor.")
    while True:
        raw = input("Temperature [C] atau q: ").strip()
        if raw.lower() == "q":
            return
        try:
            temp = float(raw)
            vib = float(input("Vibration [mm/s]: ").strip())
            if not (20 <= temp <= 100 and 0 <= vib <= 12):
                print("Input di luar rentang simulasi (T:20..100; V:0..12)")
                continue
            k = int(input("K ganjil [1/3/5/7]: ").strip())
            pred, neighbors = predict_knn(temp, vib, k)
            print("Prediction:", LABELS[pred])
            print("Nearest neighbors (jarak kuadrat, indeks, kelas):")
            for d, idx, label in neighbors:
                print("  %.5f | %d | %s" % (d, idx, LABELS[label]))
        except (ValueError, TypeError) as e:
            print("Input tidak valid:", e)


def benchmark(repetitions=100):
    # Menghindari waktu input/print dalam timing; tanpa dataset generator.
    samples = [(39, 3.1), (57, 5.5), (70, 7.5), (47, 6.5)]
    gc.collect()
    start_free = gc.mem_free()
    for temp, vib in samples:  # warm-up
        predict_knn(temp, vib, 3)
    elapsed = []
    for j in range(repetitions):
        temp, vib = samples[j % len(samples)]
        start = time.ticks_us()
        predict_knn(temp, vib, 3)
        elapsed.append(time.ticks_diff(time.ticks_us(), start))
    elapsed.sort()
    print("Inference median:", elapsed[len(elapsed) // 2], "us")
    print("Free memory before/after:", start_free, gc.mem_free(), "bytes")
    print("Note: these are approximate MicroPython-level measurements")


if __name__ == "__main__":
    demo()
    # Untuk eksperimen lain, panggil interactive() atau benchmark()
```

### Apa yang dilakukan program?

1. `TRAIN` disimpan sebagai data referensi KNN; `TEST` disimpan terpisah untuk evaluasi.
2. `predict_knn()` menghitung jarak kuadrat ke setiap data training, mengurutkan jarak, dan melakukan voting.
3. `evaluate()` membandingkan prediksi dengan label test sehingga menghasilkan accuracy dan confusion matrix.
4. `demo()` menjalankan eksperimen awal secara otomatis. `interactive()` dan `benchmark()` dipanggil saat dibutuhkan.

**JAWBAN**

MPY: soft reboot

TRAIN = 36 TEST = 12

Rentang training: T=29.0..78.8; V=1.0..9.0

K=1, scaled=True | accuracy=83.33% (10/12) | TP=5 TN=5 FP=0 FN=2

K=3, scaled=True | accuracy=75.00% (9/12) | TP=4 TN=5 FP=0 FN=3

K=5, scaled=True | accuracy=83.33% (10/12) | TP=5 TN=5 FP=0 FN=2

K=7, scaled=True | accuracy=75.00% (9/12) | TP=4 TN=5 FP=0 FN=3

Pengaruh scaling untuk K=3

K=3, scaled=False | accuracy=83.33% (10/12) | TP=5 TN=5 FP=0 FN=2

K=3, scaled=True | accuracy=75.00% (9/12) | TP=4 TN=5 FP=0 FN=3

## 7. Eksperimen A — KNN berjalan di ESP32 (15 menit)

Jalankan `main.py`. Amati keluaran `TRAIN`, `TEST`, dan accuracy untuk K = 1, 3, 5, 7.

**Target pemeriksaan dengan dataset dan kode yang tidak diubah:**

| K | Jumlah benar | Accuracy |
|---:|---:|---:|
| 1 | 10 dari 12 | 83,33% |
| 3 | 9 dari 12 | 75,00% |
| 5 | 10 dari 12 | 83,33% |
| 7 | 9 dari 12 | 75,00% |

Angka di atas merupakan hasil acuan pada Python untuk dataset ini; output MicroPython seharusnya sesuai selama logikanya sama. **K lebih besar tidak otomatis memberi akurasi lebih tinggi.**

**JAWBAN**

=== EKSPERIMEN A: KNN pada ESP32 ===

Jumlah TRAIN: 36 | Jumlah TEST: 12

Rentang Normalisasi: T=[29.0..78.8], V=[1.0..9.0]

--- Hasil Pengujian Variasi Nilai K (Scaled = True) ---

K=1 | Accuracy=83.33% (10/12) | TP=5, TN=5, FP=0, FN=2

K=3 | Accuracy=75.00% (9/12) | TP=4, TN=5, FP=0, FN=3

K=5 | Accuracy=83.33% (10/12) | TP=5, TN=5, FP=0, FN=2

K=7 | Accuracy=75.00% (9/12) | TP=4, TN=5, FP=0, FN=3


## 8. Eksperimen B — Input interaktif melalui Thonny (15 menit)

Di bagian akhir program, ubah:

```python
if __name__ == "__main__":
    demo()
```

menjadi:

```python
if __name__ == "__main__":
    interactive()
```

Jalankan, lalu masukkan temperatur, getaran, dan K melalui Shell Thonny. Coba input berikut:

| Temperatur (°C) | Getaran (mm/s) | K |
|---:|---:|---:|
| 32 | 1.6 | 3 |
| 70 | 7.5 | 3 |
| 57 | 5.5 | 3 |

Perhatikan tiga tetangga terdekat. Tuliskan alasan prediksi berdasarkan **voting**, bukan asumsi ambang temperatur atau getaran.

**JAWBAN**

PY: soft reboot

=== EKSPERIMEN B: Mode Interaktif KNN ===

Ketik q pada input temperature untuk keluar.

Temperature [C] atau q: 32

Vibration [mm/s]: 1.6

K ganjil [1/3/5/7]: 3

Hasil Prediksi : NORMAL

Nearest neighbors (jarak kuadrat | indeks TRAIN | label):

  d^2=0.00925 | Indeks #0 (T=29.0, V=1.0) | NORMAL
  
  d^2=0.01316 | Indeks #35 (T=34.8, V=2.4) | NORMAL
  
  d^2=0.02516 | Indeks #20 (T=39.8, V=1.8) | NORMAL
  
  Uji 1: Temperatur 32°C, Getaran 1,6 mm/s (K=3)3 
  
  -Tetangga Terdekat: Indeks #0 (NORMAL), Indeks #35 (NORMAL), dan Indeks #20 (NORMAL)
  
  -Alasan Voting: Seluruh tetangga terdekat (3 dari 3) memiliki label NORMAL (3\text{ vote NORMAL} vs 0\text{ vote WARNING}). Karena suara NORMAL mendominasi secara mutlak, keputusan akhir menetapkan kelas NORMAL.

Temperature [C] atau q: 70

Vibration [mm/s]: 7.5

K ganjil [1/3/5/7]: 3

Hasil Prediksi : WARNING

Nearest neighbors (jarak kuadrat | indeks TRAIN | label):

  d^2=0.02917 | Indeks #19 (T=76.4, V=8.4) | WARNING
  
  d^2=0.02976 | Indeks #18 (T=63.0, V=6.7) | WARNING
  
  d^2=0.03224 | Indeks #3 (T=68.0, V=6.1) | WARNING
  
  Uji 2: Temperatur 70°C, Getaran 7,5 mm/s (K=3)3 
  
  -Tetangga Terdekat: Indeks #19 (WARNING), Indeks #18 (WARNING), dan Indeks #3 (WARNING)
  
  -Alasan Voting: Seluruh tetangga terdekat (3 dari 3) memiliki label WARNING (3\text{ vote WARNING} vs 0\text{ vote NORMAL}). Karena jumlah vote WARNING melebihi ambang mayoritas (3 > \lfloor 3/2 \rfloor), keputusan akhir menetapkan kelas WARNING.

Temperature [C] atau q: 57

Vibration [mm/s]: 5.5

K ganjil [1/3/5/7]: 3

Hasil Prediksi : NORMAL

Nearest neighbors (jarak kuadrat | indeks TRAIN | label):

  d^2=0.01941 | Indeks #17 (T=50.8, V=5.0) | NORMAL
  
  d^2=0.01949 | Indeks #2 (T=55.8, V=4.4) | NORMAL
  
  d^2=0.02986 | Indeks #22 (T=65.4, V=5.2) | WARNING
  
  Uji 3: Temperatur 57°C, Getaran 5,5 mm/s (K=3)3 
  
  - Tetangga Terdekat: Indeks #17 (NORMAL), Indeks #2 (NORMAL), dan Indeks #22 (WARNING).
  
  - Alasan Voting: Terjadi perbedaan label di antara tetangga terdekat, yaitu 2\text{ vote NORMAL} (indeks #17 dan #2) dan 1\text{ vote WARNING} (indeks #22). Karena kelas NORMAL memenangkan suara mayoritas (2 dari 3 vote), keputusan akhir voting menetapkan kelas NORMAL.
-----------------------------------------


## 9. Eksperimen C — Pengaruh K dan normalisasi (20 menit)

Pertama, panggil `demo()` lagi dan amati output `scaled=False` dibandingkan `scaled=True`. Kedua, untuk satu input yang sama, uji K = 1, 3, 5, dan 7 melalui `interactive()`.

**Pertanyaan investigasi:** Mengapa fitur temperatur bisa terlalu dominan saat jarak dihitung tanpa scaling? Apakah seluruh input selalu berubah kelas sesudah scaling?

Untuk memperdalam, catat minimal satu contoh yang prediksinya berubah ketika K atau status scaling diubah. Jika tidak menemukan perubahan pada input awal, coba nilai yang lebih dekat dengan batas kelas seperti temperatur 45–65°C dan getaran 4–7 mm/s.

**JAWBAN**

MPY: soft reboot

=== EKSPERIMEN C: Pengaruh Scaling & Nilai K ===

1. Perbandingan Scaling pada Test Set (K=3):

K=3 | Scaled=True  | Accuracy: 75.00% (9/12)

K=3 | Scaled=False | Accuracy: 83.33% (10/12)

2. Perubahan Prediksi Akibat Scaling (K=3):

Input T=60.0, V=4.5 | Label Aktual: WARNING

  - Scaled=True  -> Prediksi: WARNING
  
  - Scaled=False -> Prediksi: NORMAL

3. Perubahan Prediksi Akibat Variasi K (T=57.0, V=5.5):
  - K=1 (Scaled=True) -> Prediksi: NORMAL
  
  - K=3 (Scaled=True) -> Prediksi: NORMAL
  
  - K=5 (Scaled=True) -> Prediksi: WARNING
  
  - K=7 (Scaled=True) -> Prediksi: WARNING

1. Mengapa Temperatur Mendominasi Tanpa Scaling?

- Rentang temperatur (DeltaT = 49,8 Celcius) jauh lebih besar dibanding rentang getaran (DeltaV = 8,0mm/s).  
 
- Pada perhitungan Euclidean tanpa scaling, selisih temperatur sebesar 5 Celcius menghasilkan kuadrat 25, sedangkan selisih getaran 1 mm/s hanya menghasilkan 1. Akibatnya, bobot getaran terabaikan.

2. Contoh Perubahan Prediksi Akibat Scaling (K=3)

- Input T=60,0 Celcius, V=4,5 mm/s (Aktual: WARNING):

- Scaled = True -- WARNING (Benar)  

- Scaled = False -- NORMAL (Salah, karena jarak temperatur mendominasi)

3. Perubahan Prediksi Akibat Variasi K (T=57,0 Celcius, V=5,mm/s)

- K=1-- NORMAL

- K=3-- NORMAL

- K=5-- WARNING

- K=7-- WARNING

Saat K bertambah besar, tetangga yang diambil makin luas sehingga mencakup lebih banyak sampel berlabel WARNING dari area sekitarnya, yang akhirnya mengubah hasil mayoritas voting.

## 10. Eksperimen D — Evaluasi pada data yang belum pernah dilihat (15 menit)

Jalankan:

```python
evaluate(k=3, scaled=True, verbose=True)
```

Perhatikan confusion matrix biner yang menggunakan **WARNING sebagai kelas positif**:

| | Prediksi WARNING (1) | Prediksi NORMAL (0) |
|---|---|---|
| Aktual WARNING (1) | TP | FN |
| Aktual NORMAL (0) | FP | TN |

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Bandingkan hasil accuracy dengan angka acuan pada Eksperimen A. Jelaskan mengapa hasil evaluasi tidak boleh hanya dihitung pada `TRAIN`.

**JAWBAN**


1. DATA KELUARAN AKTUAL SHELL THONNY (ESP32)

--------------------------------------------------------------------------------

Konfigurasi Uji: K = 3, Scaled = True (Min-Max Normalization)



Rincian Prediksi per Sampel (12 Sampel Uji):

1. T=32.0, V=1.6 | actual=NORMAL   predicted=NORMAL   [Benar  - TN]

2. T=39.0, V=3.1 | actual=NORMAL   predicted=NORMAL   [Benar  - TN]

3. T=52.0, V=3.0 | actual=NORMAL   predicted=NORMAL   [Benar  - TN]

4. T=60.0, V=4.5 | actual=WARNING  predicted=WARNING  [Benar  - TP]

5. T=47.0, V=6.5 | actual=WARNING  predicted=NORMAL   [SALAH  - FN]

6. T=70.0, V=7.5 | actual=WARNING  predicted=WARNING  [Benar  - TP]

7. T=78.0, V=8.0 | actual=WARNING  predicted=WARNING  [Benar  - TP]

8. T=35.0, V=7.9 | actual=NORMAL   predicted=NORMAL   [Benar  - TN]

9. T=73.0, V=2.5 | actual=WARNING  predicted=NORMAL   [SALAH  - FN]

10. T=57.0, V=5.5 | actual=WARNING  predicted=NORMAL   [SALAH  - FN]

11. T=46.0, V=4.4 | actual=NORMAL   predicted=NORMAL   [Benar  - TN]

12. T=65.0, V=6.2 | actual=WARNING  predicted=WARNING  [Benar  - TP]



Ringkasan Evaluasi:

- Total Sampel Uji    : 12

- Sampel Benar         : 9

- Sampel Salah         : 3

- Accuracy             : 75.00% (9/12)

- True Positive (TP)   : 4

- True Negative (TN)   : 5

- False Positive (FP)  : 0

- False Negative (FN)  : 3


================================================================================

2. TABEL CONFUSION MATRIX (KELAS POSITIF = WARNING)

================================================================================



                 | Prediksi WARNING (1) | Prediksi NORMAL (0) | Total Aktual

-----------------+----------------------+---------------------+-------------

Aktual WARNING (1)| TP = 4               | FN = 3              | 7 Sampel

Aktual NORMAL (0) | FP = 0               | TN = 5              | 5 Sampel

-----------------+----------------------+---------------------+-------------

Total Prediksi   | 4 Sampel             | 8 Sampel            | 12 Sampel



Penjelasan Komponen:

1. True Positive (TP = 4):

   Kondisi anomali/bahaya mesin yang tepat terdeteksi sebagai WARNING.

   Sampel:

   - T=60.0, V=4.5

   - T=70.0, V=7.5

   - T=78.0, V=8.0

   - T=65.0, V=6.2



2. True Negative (TN = 5):

   Kondisi mesin normal yang tepat terdeteksi sebagai NORMAL.

   Sampel:

   - T=32.0, V=1.6

   - T=39.0, V=3.1

   - T=52.0, V=3.0

   - T=35.0, V=7.9

   - T=46.0, V=4.4



3. False Positive (FP = 0):

   Kondisi mesin normal tetapi salah dideteksi sebagai bahaya (alarm palsu).

   Nilai FP = 0 membuktikan model memiliki spesifisitas sempurna (100%), artinya

   sistem tidak pernah membunyikan alarm palsu untuk mesin yang sebenarnya aman.



4. False Negative (FN = 3):

   Kondisi mesin berbahaya tetapi luput dan terprediksi aman (NORMAL).

   Sampel yang mengalami kegagalan:

   - T=47.0, V=6.5 (Getaran tinggi tetapi suhu sedang)

   - T=73.0, V=2.5 (Suhu tinggi tetapi getaran rendah)

   - T=57.0, V=5.5 (Titik transisi tepat di perbatasan kelas)



Perhitungan Akurasi:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

         = (4 + 5) / (4 + 5 + 0 + 3)

         = 9 / 12

         = 75.00%





================================================================================

3. JAWABAN PERTANYAAN ANALISIS MODUL (EKSPERIMEN D)

================================================================================



Pertanyaan:

Bandingkan hasil accuracy dengan angka acuan pada Eksperimen A. Jelaskan mengapa

hasil evaluasi model machine learning TIDAK BOLEH hanya dihitung pada data training?



Jawaban:



1. Perbandingan dengan Eksperimen A:

   Pada Eksperimen A, angka acuan untuk K=3 dengan scaling adalah 75.00% (9 benar

   dari 12 data). Hasil pengujian langsung pada ESP32 menunjukkan nilai yang persis

   sama, yaitu akurasi 75.00% dengan TP=4, TN=5, FP=0, dan FN=3. Hal ini membuktikan

   bahwa algoritma klasifikasi KNN pada MicroPython berjalan konsisten dan deterministik.



2. Alasan Mengapa Evaluasi Tidak Boleh Hanya pada Data Training:

   a. Menghindari Bias Hafalan (Overfitting):

      Algoritma KNN tidak mempelajari bobot matematis melainkan menghafal seluruh

      titik data referensi. Jika diuji menggunakan data training itu sendiri dengan

      nilai K=1, jarak terdekat yang ditemukan pasti adalah sampel dirinya sendiri

      (jarak = 0). Ini akan menghasilkan akurasi semu 100%, yang membuat model

      terlihat sempurna padahal sebenarnya hanya menghafal data lampau.



   b. Menguji Kemampuan Generalisasi:

      Tujuan utama penanaman machine learning pada mikrokontroler (Edge AI/ESP32)

      adalah memprediksi kondisi operasional mesin di masa depan yang polanya belum

      pernah tersimpan di memori. Pengujian menggunakan TEST set terpisah adalah

      satu-satunya cara objektif untuk mengukur daya generalisasi model.



   c. Mendeteksi Kelemahan Kritis di Dunia Nyata (False Negative):

      Jika evaluasi hanya dilakukan pada data training, kelemahan pada area perbatasan

      tidak akan pernah terdeteksi. Melalui evaluasi TEST set ini, terbukti ada 3

      kondisi mesin bahaya yang luput (FN=3). Pada sistem predictive maintenance pabrik,

      False Negative jauh lebih fatal daripada False Positive karena dapat menyebabkan

      kerusakan mesin total tanpa peringatan dini.

## 11. Eksperimen E — Waktu inference pada ESP32 (15 menit)

Panggil `benchmark(100)` dari Shell Thonny atau ubah pemanggilan pada bagian `__main__`. Fungsi ini:

- Menjalankan pemanasan ringan sebelum pencatatan waktu.
- Mengukur 100 kali waktu eksekusi `predict_knn()` dengan `time.ticks_us()`.
- Melaporkan **median** waktu inference dan memori bebas (`gc.mem_free()`).

**Penting tentang metrik:** median waktu adalah perkiraan dari pengukuran MicroPython, bukan latency real-time deterministik. Memori bebas sebelum/sesudah benchmark **bukan** ukuran persis total konsumsi RAM model; alokasi sementara dan *garbage collection* dapat memengaruhinya. Untuk membandingkan ukuran dataset, gunakan prosedur yang sama dan simpan jumlah titik, latency median, serta memori bebas secara konsisten.

**JAWBAN**

=== EKSPERIMEN E: Benchmark Inference Latency & Memori ===

Jumlah Data TRAIN : 36 sampel

Jumlah Pengujian  : 100 kali

Inference Median  : 16620 us (mikrodetik)

Memori Bebas (sebelum/sesudah) : 160352 / 125952 bytes

### Tantangan lanjutan (opsional)

Salin beberapa baris `TRAIN` untuk membuat dataset referensi lebih besar, misalnya 72 atau 144 sampel. Ulangi benchmark dengan input dan jumlah repetisi yang sama. Ingat, duplikasi data dapat memengaruhi voting dan **bukan** penambahan informasi baru sehingga jangan menafsirkan peningkatan accuracy sebagai akibat data lebih bermutu.

**JAWBAN**

=== TANTANGAN LANJUTAN: Skalabilitas Dataset pada ESP32 ===

Ukuran Dataset TRAIN : 36 sampel (1x)

Median Inference     : 16419 us (16.42 ms)

Memori Bebas (S/A)   : 160016 / 125616 bytes

Alokasi Heap 100 Run : 34400 bytes

----------------------------------------

Ukuran Dataset TRAIN : 72 sampel (2x)

Median Inference     : 39519 us (39.52 ms)

Memori Bebas (S/A)   : 158992 / 130640 bytes

Alokasi Heap 100 Run : 28352 bytes

-----------------------------------------

Ukuran Dataset TRAIN : 144 sampel (4x)

Median Inference     : 109738 us (109.74 ms)

Memori Bebas (S/A)   : 158704 / 114112 bytes

Alokasi Heap 100 Run : 44592 bytes

## 12. Lembar hasil mahasiswa

| Variasi | K | Scaling | Accuracy (%) | Median inference (µs) | Catatan |
|---|---:|---|---:|---:|---|
| A | 1 | Ya | | | |
| B | 3 | Ya | | | |
| C | 5 | Ya | | | |
| D | 7 | Ya | | | |
| E | 3 | Tidak | | | |

**JAWABAN**

| Variasi | K | Scaling | Accuracy (%) | Median inference (µs) | Catatan |
|---|---:|---|---:|---:|---|
| A | 1 | Ya |83.33% |15509 us |10/12 benar; batas keputusan tajam |
| B | 3 | Ya | 75.00%|15532 us |9/12 benar; acuan eksperimen utama |
| C | 5 | Ya |83.33% |15546 us |10/12 benar; voting lebih stabil |
| D | 7 | Ya |75.00% |15581 us |9/12 benar; batas keputusan melunak |
| E | 3 | Tidak |83.33% |14210 us |10/12 benar; suhu mendominasi jarak |

## 13. Pertanyaan analisis

1. Mengapa KNN tidak memerlukan proses pembaruan koefisien seperti regresi linear, tetapi tetap membutuhkan data training?
2. Mengapa normalisasi dapat mengubah prediksi KNN? Jelaskan menggunakan skala temperatur dan getaran.
3. Apakah K yang lebih besar selalu menghasilkan accuracy lebih tinggi? Gunakan hasil percobaan.
4. Apa perbedaan waktu inference dan waktu input melalui terminal serial? Mengapa *timing* tidak dilakukan saat `input()` atau `print()`?
5. Jika data training diperbesar 10 kali, apa yang kemungkinan terjadi pada penggunaan memori dan waktu inference di ESP32? Apa yang perlu diukur untuk membuktikannya?

**JAWBAN**
1. Alasan KNN Tanpa Pembaruan Koefisien tetapi Butuh Data Training
KNN tergolong algoritma instance-based atau lazy learning. KNN tidak melakukan fase pelatihan matematis untuk mencari bobot/koefisien tetap seperti regresi linear. Sebagai gantinya, KNN menunda seluruh proses hitungan sampai ada data baru yang ingin diprediksi. Data training tetap wajib disimpan di memori ESP32 sebagai titik referensi mentah untuk dihitung jaraknya satu per satu saat inference berlangsung.

2. Mengapa Normalisasi Mengubah Prediksi (Skala Temperatur vs Getaran)
Normalisasi menyetarakan rentang nilai antar-fitur. Tanpa normalisasi, perhitungan jarak Euclidean didominasi oleh fitur dengan skala nilai yang jauh lebih besar. Pada dataset ini, temperatur punya rentang jauh lebih lebar (49,8Celcius) dibanding getaran (8,0mm/s). Selisih temperatur 5 Celcius menghasilkan nilai kuadrat 25, sedangkan selisih getaran 1 mm/s cuma menghasilkan 1. Tanpa scaling, pengaruh getaran tertutup oleh temperatur. Setelah di-scale ke rentang [0, 1], kedua fitur punya bobot yang seimbang.

3. Pengaruh Nilai K Terhadap Akurasi
K yang lebih besar tidak selalu menghasilkan akurasi lebih tinggi. Pada pengujian data uji:   
- K=1 akurasi 83,33%   
- K=3 akurasi 75,00%   
- K=5 akurasi 83,33%   
- K=7 akurasi 75,00%  
Nilai K yang terlalu kecil $K=1) rentan terhadap noise, sedangkan K yang terlalu besar (K=7) memicu underfitting karena mengambil tetangga yang lokasinya terlalu jauh hingga mengaburkan batas antarkelas.

4. Perbedaan Waktu Inference vs Waktu Input SerialWaktu Inference: 

Waktu murni yang dipakai prosesor ESP32 untuk mengeksekusi algoritma KNN (menghitung jarak kuadrat, mengurutkan array, dan voting).   Waktu Input Serial: Waktu komunikasi hardware serial (UART) ditambah jeda tunggu pengetikan manusia.   Timing diisolasi dari input() atau print() karena proses I/O dan interaksi manusia memakan waktu dalam orde milidetik hingga detik ($10^3 - 10^6\ \mu\text{s}$). Angka tersebut akan menutupi waktu eksekusi KNN asli yang hanya butuh hitungan mikrodetik ($\mu\text{s}$).

5. Proyeksi Pembesaran Data Training 10 Kali Lipat
- Waktu Inference: Bakal membengkak sekitar 10 kali lipat. Kompleksitas waktu KNN bersifat linier $\mathcal{O}(N)$, sehingga semakin banyak titik sampel ($N$), semakin banyak perhitungan jarak yang harus dilakukan.   

- Penggunaan Memori: Konsumsi RAM heap meningkat tajam karena array data training dan list penampung jarak sementara membesar 10 kali lipat, yang berisiko memicu MemoryError pada ESP32.   

- Yang perlu diukur untuk membuktikannya:
	1 Waktu eksekusi murni pakai time.ticks_us() 
	2 sebelum dan sesudah pemanggilan predict_knn().   Sisa RAM bebas menggunakan gc.mem_free() sebelum dan sesudah array diperbesar serta fungsi dijalankan.   

## 14. Kesimpulan dan tugas

**Kesimpulan:** KNN dapat ditulis dengan MicroPython dan dijalankan seluruhnya pada ESP32 tanpa sensor atau library machine learning. Parameter preprocessing dan pemisahan data training/test tetap penting meskipun perangkat sangat sederhana.

**Pengumpulan:** serahkan `main.py`, foto/screenshot output Shell, tabel eksperimen, dan jawaban pertanyaan analisis. Jangan menyerahkan accuracy training sebagai pengganti accuracy test.

---

**Catatan instruktur:** seluruh hasil acuan diuji pada Python standar dengan algoritma identik; performa waktu/memori perlu diukur pada board mahasiswa. Tidak ada hardware sensor maupun koneksi cloud yang terlibat.
