# 📊 OUTLINE SLIDE PRESENTASI UAS
## CALORIE COUNTER AI

---

## SLIDE 1: COVER
**Desain:** Background gradient hijau, logo aplikasi (emoji 🍎)

```
CALORIE COUNTER AI
Aplikasi Penghitung Kalori Makanan Indonesia
Berbasis Kecerdasan Buatan

Nama: [NAMA ANDA]
NIM: [NIM ANDA]
Mata Kuliah: [NAMA MATA KULIAH]
Dosen Pengampu: [NAMA DOSEN]

[LOGO UNIVERSITAS]
```

---

## SLIDE 2: DAFTAR ISI

```
📋 AGENDA PRESENTASI

1. Latar Belakang
2. Tujuan & Manfaat
3. Fitur Aplikasi
4. Demo Live
5. Arsitektur Sistem
6. Teknologi yang Digunakan
7. Dataset & Training
8. Hasil Evaluasi
9. Kelebihan & Kekurangan
10. Kesimpulan & Pengembangan
```

---

## SLIDE 3: LATAR BELAKANG

```
🎯 LATAR BELAKANG

MASALAH:
❌ Sulit menghitung kalori makanan secara manual
❌ Informasi nutrisi makanan Indonesia terbatas
❌ Tracking kalori memakan waktu lama
❌ Kesadaran pola makan sehat meningkat

SOLUSI:
✅ Aplikasi AI untuk deteksi makanan otomatis
✅ Database kalori makanan Indonesia
✅ Interface modern dan mudah digunakan
✅ Perhitungan kalori real-time
```

**Visual:** Icon masalah di kiri, icon solusi di kanan dengan panah

---

## SLIDE 4: TUJUAN & MANFAAT

```
🎯 TUJUAN PROJECT

1. Memudahkan pengguna menghitung kalori makanan
2. Implementasi Deep Learning untuk klasifikasi gambar
3. Menyediakan database kalori makanan Indonesia
4. Memberikan UX yang intuitif dan modern

💡 MANFAAT

Untuk Pengguna:
• Tracking kalori lebih mudah dan cepat
• Mendukung pola hidup sehat
• Edukasi nutrisi makanan

Untuk Pengembang:
• Pembelajaran implementasi AI
• Portfolio project
• Kontribusi teknologi kesehatan
```

---

## SLIDE 5: FITUR APLIKASI

```
✨ FITUR UTAMA

🖼️ UPLOAD & ANALISIS FOTO
   Upload gambar makanan dan dapatkan hasil instan

🤖 3 MODEL AI
   • MobileNetV2 - Cepat & Efisien
   • EfficientNetB0 - Akurasi Tinggi
   • CNN Sederhana - Model Dasar

⚖️ PERHITUNGAN DINAMIS
   Sesuaikan porsi untuk kalori akurat

📊 RIWAYAT MAKANAN
   Simpan dan kelola history analisis

🍛 18 JENIS MAKANAN
   Rendang, Nasi Goreng, Sate, Gudeg, dll.
```

**Visual:** Screenshot fitur-fitur aplikasi

---

## SLIDE 6: MAKANAN YANG DIDUKUNG

```
🍽️ 18 JENIS MAKANAN INDONESIA

1. Ayam Goreng        10. Gulai Tunjang
2. Ayam Pop           11. Nasi Goreng
3. Bakso              12. Pempek
4. Bebek Betutu       13. Rawon
5. Dendeng Batokok    14. Rendang
6. Gado-Gado          15. Sate
7. Gudeg              16. Soto
8. Gulai Ikan         17. Telur Balado
9. Gulai Tambusu      18. Telur Dadar
```

**Visual:** Grid 3x6 dengan foto masing-masing makanan

---

## SLIDE 7: DEMO LIVE

```
🎬 DEMO APLIKASI

[LIVE DEMONSTRATION]

Langkah-langkah:
1. Buka aplikasi di browser
2. Pilih model AI
3. Upload foto makanan
4. Lihat proses analisis
5. Hasil deteksi & kalori
6. Sesuaikan porsi
7. Lihat riwayat
```

**Catatan:** Slide ini untuk transisi ke demo live

---

## SLIDE 8: ARSITEKTUR SISTEM

```
🏗️ ARSITEKTUR SISTEM

┌─────────────┐
│   USER      │
│  (Browser)  │
└──────┬──────┘
       │ HTTP Request
       ▼
┌─────────────────┐
│  FLASK SERVER   │
│  - Routing      │
│  - File Upload  │
│  - API Endpoint │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  PREPROCESSING  │
│  - Resize       │
│  - Normalize    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   AI MODEL      │
│  - MobileNet    │
│  - EfficientNet │
│  - SimpleCNN    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ CALORIE DB      │
│ - Lookup        │
│ - Calculate     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  JSON RESPONSE  │
│  - Food name    │
│  - Calories     │
│  - Confidence   │
└─────────────────┘
```

---

## SLIDE 9: TEKNOLOGI YANG DIGUNAKAN

```
💻 TECH STACK

BACKEND:
🐍 Python 3.x
🌶️ Flask - Web Framework
🧠 TensorFlow/Keras - Deep Learning
📊 NumPy - Array Operations

FRONTEND:
🌐 HTML5 - Structure
🎨 Tailwind CSS - Styling
⚡ JavaScript - Interactivity
📱 Responsive Design

MACHINE LEARNING:
🤖 MobileNetV2 (Google)
🎯 EfficientNetB0 (Google)
🔧 Custom CNN
📦 Transfer Learning
```

**Visual:** Logo masing-masing teknologi

---

## SLIDE 10: DATASET & TRAINING

```
📚 DATASET

• 18 Kelas Makanan Indonesia
• ~500-1000 gambar per kelas
• Total: ~10,000+ gambar
• Split: 80% Training, 20% Validation

🎓 TRAINING PROCESS

1. Data Collection & Labeling
2. Data Augmentation
   - Rotation, Flip, Zoom
   - Brightness, Contrast
3. Transfer Learning
   - Pre-trained ImageNet weights
   - Fine-tuning last layers
4. Evaluation
   - Accuracy, Precision, Recall
   - Confusion Matrix
```

**Visual:** Diagram proses training

---

## SLIDE 11: PREPROCESSING & AUGMENTATION

```
🔧 IMAGE PREPROCESSING

Input Image (Any Size)
        ↓
Resize to Model Size
• MobileNet: 128x128
• EfficientNet: 224x224
        ↓
Normalize Pixels (0-1)
        ↓
Expand Dimensions
        ↓
Ready for Prediction

📈 DATA AUGMENTATION

• Rotation: ±20°
• Width/Height Shift: 20%
• Zoom: 20%
• Horizontal Flip
• Brightness: ±20%
```

**Visual:** Before/After augmentation examples

---

## SLIDE 12: MODEL COMPARISON

```
⚖️ PERBANDINGAN MODEL

┌──────────────┬─────────┬──────────┬─────────┐
│    Model     │ Akurasi │   Speed  │   Size  │
├──────────────┼─────────┼──────────┼─────────┤
│ MobileNetV2  │  ~85%   │  ⚡⚡⚡   │  ~14MB  │
│ EfficientNet │  ~90%   │  ⚡⚡     │  ~29MB  │
│ SimpleCNN    │  ~75%   │  ⚡⚡⚡⚡ │  ~5MB   │
└──────────────┴─────────┴──────────┴─────────┘

REKOMENDASI:
✅ MobileNetV2 - Best balance (speed + accuracy)
✅ EfficientNet - Maximum accuracy
✅ SimpleCNN - Learning purposes
```

**Visual:** Bar chart comparison

---

## SLIDE 13: CARA KERJA PREDIKSI

```
🔮 PREDICTION FLOW

1. Load Image
   img = load_img(path, target_size=(128,128))

2. Convert to Array
   img_array = img_to_array(img)

3. Normalize
   img_array = img_array / 255.0

4. Expand Dimensions
   img_array = np.expand_dims(img_array, 0)

5. Predict
   predictions = model.predict(img_array)

6. Get Class
   class_idx = np.argmax(predictions[0])
   confidence = np.max(predictions[0]) * 100

7. Map to Food Name
   food_name = labels[class_idx]

8. Lookup Calories
   calories = calorie_db[food_name]
```

---

## SLIDE 14: DATABASE KALORI

```
📊 CALORIE DATABASE (per 100g)

Makanan Tinggi Kalori:
🔥 Dendeng Batokok: 300 kal
🔥 Ayam Goreng: 275 kal
🔥 Nasi Goreng: 267 kal

Makanan Sedang:
🟡 Bebek Betutu: 250 kal
🟡 Sate: 200 kal
🟡 Ayam Pop: 200 kal

Makanan Rendah Kalori:
🟢 Gulai Ikan: 106 kal
🟢 Gado-Gado: 137 kal
🟢 Telur Balado: 142 kal

Sumber Data:
• Database Nutrisi Kemenkes RI
• USDA Food Database
• Penelitian Ilmiah
```

**Visual:** Bar chart kalori per makanan

---

## SLIDE 15: FITUR UNGGULAN

```
⭐ KEUNGGULAN APLIKASI

1. 🎨 UI/UX MODERN
   • Clean & Minimalist Design
   • Smooth Animations
   • Responsive Layout

2. ⚡ PERFORMA CEPAT
   • Model Caching
   • Optimized Preprocessing
   • 2-3 detik per analisis

3. 🎯 AKURAT
   • State-of-the-art Models
   • Confidence Score Display
   • Multiple Model Options

4. 📱 USER-FRIENDLY
   • Drag & Drop Upload
   • Real-time Feedback
   • Auto-save History

5. 🔧 FLEKSIBEL
   • Dynamic Portion Calculation
   • Editable History
   • Export-ready Data
```

---

## SLIDE 16: SCREENSHOT APLIKASI

```
📸 TAMPILAN APLIKASI

[4 Screenshot dalam grid 2x2]

1. Home Page
   - Upload area
   - Model selection

2. Loading Screen
   - Progress bar
   - Loading stages

3. Result Page
   - Food image
   - Calorie info
   - Portion input

4. History Page
   - List of analyses
   - Delete options
```

**Visual:** Screenshot actual aplikasi

---

## SLIDE 17: HASIL EVALUASI

```
📈 EVALUATION RESULTS

METRICS (MobileNetV2):
✅ Training Accuracy: ~88%
✅ Validation Accuracy: ~85%
✅ Precision: ~84%
✅ Recall: ~83%
✅ F1-Score: ~83%

CONFUSION MATRIX:
[Tampilkan confusion matrix heatmap]

TOP 5 BEST DETECTED:
1. Rendang - 95% avg confidence
2. Nasi Goreng - 92%
3. Sate - 90%
4. Bakso - 88%
5. Pempek - 87%

CHALLENGING CASES:
⚠️ Ayam Goreng vs Ayam Pop
⚠️ Gulai variants
```

**Visual:** Confusion matrix + bar charts

---

## SLIDE 18: KELEBIHAN

```
✅ KELEBIHAN APLIKASI

🎯 FUNGSIONAL
• Deteksi makanan otomatis & akurat
• 3 pilihan model AI
• Perhitungan kalori dinamis
• Riwayat tersimpan otomatis

💎 TEKNIS
• Clean code architecture
• Modular design
• Scalable system
• Well-documented

🎨 UI/UX
• Modern & attractive design
• Intuitive interface
• Smooth animations
• Responsive layout

🚀 PERFORMA
• Fast inference (2-3s)
• Model caching
• Optimized preprocessing
```

---

## SLIDE 19: KEKURANGAN & SOLUSI

```
⚠️ KEKURANGAN & RENCANA PERBAIKAN

KEKURANGAN                    SOLUSI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔸 Hanya 18 jenis makanan  →  Tambah dataset
                              Target: 50+ makanan

🔸 Perlu koneksi internet  →  Develop mobile app
                              dengan TFLite (offline)

🔸 Estimasi porsi manual   →  Tambah AI portion
                              estimation

🔸 Hanya info kalori       →  Tambah nutrisi lengkap
                              (protein, lemak, karbo)

🔸 Single food detection   →  Multi-food detection
                              dalam satu foto

🔸 Bahasa Indonesia only   →  Multi-language support
```

---

## SLIDE 20: CHALLENGES & LESSONS LEARNED

```
🎓 PEMBELAJARAN & TANTANGAN

CHALLENGES:
❗ Dataset collection & labeling
❗ Model overfitting
❗ Similar food classification
❗ Portion size estimation
❗ Deployment optimization

LESSONS LEARNED:
💡 Transfer learning sangat efektif
💡 Data augmentation penting untuk generalisasi
💡 User feedback crucial untuk improvement
💡 Balance antara akurasi dan kecepatan
💡 Documentation saves time

SKILLS GAINED:
✅ Deep Learning (TensorFlow/Keras)
✅ Web Development (Flask)
✅ UI/UX Design
✅ Data Collection & Preprocessing
✅ Model Evaluation & Optimization
```

---

## SLIDE 21: PENGEMBANGAN MASA DEPAN

```
🚀 FUTURE DEVELOPMENT

SHORT TERM (1-3 bulan):
📱 Progressive Web App (PWA)
🍽️ Tambah 20+ jenis makanan baru
🌍 Multi-language support
📊 Export data to CSV/PDF

MEDIUM TERM (3-6 bulan):
📱 Mobile App (React Native/Flutter)
🤖 Improve model accuracy (>90%)
🍴 Multi-food detection
⚖️ AI-based portion estimation

LONG TERM (6-12 bulan):
🏥 Integration dengan health apps
👥 Social features (share meals)
📈 Personalized recommendations
🔬 Nutrition analysis (macro/micro)
🎯 Meal planning assistant
```

---

## SLIDE 22: IMPACT & CONTRIBUTION

```
🌟 DAMPAK & KONTRIBUSI

UNTUK MASYARAKAT:
✅ Memudahkan tracking kalori
✅ Edukasi nutrisi makanan Indonesia
✅ Mendukung gaya hidup sehat
✅ Accessible & free to use

UNTUK AKADEMIK:
✅ Implementasi AI dalam kesehatan
✅ Dataset makanan Indonesia
✅ Open source contribution
✅ Research opportunity

UNTUK INDUSTRI:
✅ Proof of concept AI nutrition
✅ Scalable architecture
✅ Integration-ready system
✅ Commercial potential
```

---

## SLIDE 23: KESIMPULAN

```
📝 KESIMPULAN

✅ Berhasil mengimplementasikan Deep Learning
   untuk klasifikasi makanan Indonesia

✅ Aplikasi dapat mendeteksi 18 jenis makanan
   dengan akurasi ~85%

✅ Interface modern dan user-friendly
   memudahkan pengguna

✅ Fitur perhitungan kalori dinamis
   memberikan fleksibilitas

✅ Sistem scalable dan siap dikembangkan
   lebih lanjut

🎯 TAKEAWAY:
AI dapat dimanfaatkan untuk mendukung
kesehatan masyarakat dengan cara yang
praktis dan accessible
```

---

## SLIDE 24: REFERENSI

```
📚 REFERENSI

PAPERS:
[1] Sandler et al. (2018). "MobileNetV2: Inverted Residuals 
    and Linear Bottlenecks"
[2] Tan & Le (2019). "EfficientNet: Rethinking Model Scaling 
    for Convolutional Neural Networks"
[3] Krizhevsky et al. (2012). "ImageNet Classification with 
    Deep Convolutional Neural Networks"

DATASETS:
[4] Database Komposisi Pangan Indonesia - Kemenkes RI
[5] USDA Food Composition Database
[6] Custom collected dataset (10,000+ images)

FRAMEWORKS:
[7] TensorFlow Documentation - tensorflow.org
[8] Flask Documentation - flask.palletsprojects.com
[9] Keras Applications - keras.io/api/applications

OTHERS:
[10] Various nutrition research papers
[11] Indonesian food nutrition studies
```

---

## SLIDE 25: TERIMA KASIH

```
🙏 TERIMA KASIH

Terima kasih atas perhatian
Bapak/Ibu Dosen dan teman-teman

CONTACT:
📧 Email: [email@example.com]
💻 GitHub: [github.com/username]
🔗 LinkedIn: [linkedin.com/in/username]

DEMO LINK:
🌐 [URL jika sudah deploy]

REPOSITORY:
📦 [GitHub repository link]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SESI TANYA JAWAB
❓ Questions & Answers
```

**Visual:** Logo aplikasi besar di tengah, contact info di bawah

---

## 🎨 DESIGN GUIDELINES

### Color Scheme:
- **Primary**: #4CAF50 (Green)
- **Secondary**: #388E3C (Dark Green)
- **Accent**: #C8E6C9 (Light Green)
- **Text**: #212121 (Dark Gray)
- **Background**: #FFFFFF (White)

### Fonts:
- **Headings**: Montserrat Bold
- **Body**: Open Sans Regular
- **Code**: Fira Code

### Layout:
- **Consistent margins**: 1 inch all sides
- **Logo placement**: Top right corner
- **Slide numbers**: Bottom right
- **Max bullet points**: 5-7 per slide
- **Font size**: Min 18pt for body text

---

## 📊 VISUAL ELEMENTS TO INCLUDE

1. **Screenshots** - Actual app interface
2. **Diagrams** - Architecture, flow charts
3. **Charts** - Bar charts, pie charts for metrics
4. **Icons** - Emoji or icon sets for visual appeal
5. **Code snippets** - Syntax highlighted
6. **Photos** - Sample food images
7. **Animations** - Subtle transitions (optional)

---

## ⏱️ TIMING PER SLIDE

| Slide Range | Time | Notes |
|-------------|------|-------|
| 1-2 | 1 min | Quick intro |
| 3-6 | 4 min | Background & features |
| 7 | 8 min | **DEMO LIVE** |
| 8-14 | 5 min | Technical deep dive |
| 15-19 | 3 min | Evaluation & analysis |
| 20-23 | 2 min | Future & conclusion |
| 24-25 | 1 min | References & closing |

**TOTAL: ~24 minutes** (including buffer time)

---

*Sesuaikan jumlah slide dan konten berdasarkan waktu presentasi yang tersedia*
