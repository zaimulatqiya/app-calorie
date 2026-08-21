# Aplikasi Penghitung Kalori Makanan Indonesia

Aplikasi web berbasis kecerdasan buatan untuk mengenali jenis makanan Indonesia dari foto dan menghitung estimasi kalorinya secara otomatis.

---

## Tentang Aplikasi

Aplikasi ini dibuat sebagai bagian dari penelitian skripsi tentang klasifikasi makanan Indonesia menggunakan deep learning. Pengguna cukup mengunggah foto makanan, lalu sistem akan mengenali jenis makanannya dan menampilkan estimasi kalori berdasarkan porsi standar.

Aplikasi ini mendukung **18 jenis makanan Indonesia**, yaitu:

> Ayam Goreng, Ayam Pop, Bakso, Bebek Betutu, Dendeng Batokok, Gado-Gado, Gudeg, Gulai Ikan, Gulai Tambusu, Gulai Tunjang, Nasi Goreng, Pempek, Rawon, Rendang, Sate, Soto, Telur Balado, Telur Dadar

---

## Model yang Digunakan

Tiga model deep learning dilatih dan dibandingkan dalam penelitian ini:

| Model | Arsitektur | Ukuran Input |
|---|---|---|
| SimpleCNN | CNN dari nol (3 layer konvolusi) | 128 x 128 |
| MobileNetV2 | Transfer learning dari ImageNet | 128 x 128 |
| EfficientNetB0 | Fine-tuned dari ImageNet | 224 x 224 |

Model yang aktif digunakan di aplikasi dapat dipilih langsung oleh pengguna saat melakukan analisis.

---

## Fitur Utama

- Unggah foto makanan dan dapatkan hasil klasifikasi secara instan
- Pilih model yang ingin digunakan (SimpleCNN, MobileNetV2, atau EfficientNetB0)
- Lihat tingkat kepercayaan (confidence score) dari hasil prediksi
- Hitung ulang kalori dengan mengubah berat porsi secara manual
- Simpan riwayat analisis makanan
- Hapus riwayat satu per satu atau sekaligus

---

## Struktur Folder

```
├── app.py                          # File utama aplikasi Flask
├── evaluate_models.py              # Script evaluasi dan perbandingan model
├── train_simplecnn.py              # Script pelatihan model SimpleCNN
├── train_mobilenet.py              # Script pelatihan model MobileNetV2
├── train_efficientnet_finetune.py  # Script pelatihan model EfficientNetB0
├── requirements.txt                # Daftar library yang dibutuhkan
├── Dockerfile                      # Konfigurasi Docker untuk deployment
├── templates/
│   └── index.html                  # Tampilan antarmuka web
├── static/
│   ├── logo-icon.png
│   └── logo-text.png
├── models/
│   ├── food_model.h5                         # Model SimpleCNN
│   ├── food_model_mobilenet.h5               # Model MobileNetV2
│   ├── food_model_efficientnet_finetuned.h5  # Model EfficientNetB0
│   ├── class_indices.json                    # Mapping kelas makanan
│   ├── training_history_simplecnn.json       # Riwayat training SimpleCNN
│   ├── training_history_mobilenet.json       # Riwayat training MobileNetV2
│   └── training_history_efficientnet.json    # Riwayat training EfficientNetB0
├── evaluation_results/             # Hasil evaluasi model (grafik dan CSV)
├── data/
│   └── dataset_makanan_indonesia/  # Dataset foto makanan
└── history.json                    # Riwayat analisis pengguna
```

---

## Cara Menjalankan Aplikasi

### Persyaratan

- Python 3.9 atau lebih baru
- pip (package installer Python)

### Langkah Instalasi

**1. Clone atau ekstrak project ini ke komputer Anda**

**2. Buat virtual environment (opsional tapi disarankan)**
```bash
python -m venv .venv
```

Aktifkan virtual environment:
- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

**3. Install semua library yang dibutuhkan**
```bash
pip install -r requirements.txt
```

> Proses ini membutuhkan waktu beberapa menit karena TensorFlow berukuran cukup besar.

**4. Jalankan aplikasi**
```bash
python app.py
```

**5. Buka browser dan akses:**
```
http://127.0.0.1:5000
```

---

## Cara Menggunakan Aplikasi

1. Buka aplikasi di browser
2. Pilih model yang ingin digunakan
3. Klik tombol unggah dan pilih foto makanan
4. Klik tombol **Analisis**
5. Hasil klasifikasi dan estimasi kalori akan muncul secara otomatis
6. Ubah berat porsi jika diperlukan untuk menghitung ulang kalori

---

## Menjalankan Evaluasi Model

Untuk menjalankan evaluasi perbandingan performa antar model, jalankan perintah berikut:

```bash
pip install -r requirements_evaluation.txt
python evaluate_models.py
```

Hasil evaluasi akan tersimpan secara otomatis di folder `evaluation_results/`.

---

## Deployment dengan Docker

Aplikasi ini sudah dikonfigurasi untuk deployment menggunakan Docker, khususnya untuk platform Hugging Face Spaces.

```bash
docker build -t aplikasi-kalori .
docker run -p 7860:7860 aplikasi-kalori
```

---

## Teknologi yang Digunakan

- **Backend:** Python, Flask
- **Deep Learning:** TensorFlow, Keras
- **Model:** SimpleCNN, MobileNetV2, EfficientNetB0
- **Deployment:** Docker, Gunicorn
- **Frontend:** HTML, CSS, JavaScript

---

## Catatan

- Estimasi kalori dihitung berdasarkan **porsi standar 150 gram** dan dapat diubah secara manual
- Data kalori mengacu pada nilai kalori per 100 gram untuk masing-masing jenis makanan
- Aplikasi ini dirancang untuk mendukung kesadaran gizi, bukan sebagai pengganti konsultasi ahli gizi
