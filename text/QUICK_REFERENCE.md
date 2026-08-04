# 🎴 QUICK REFERENCE CARD
## CALORIE COUNTER AI - Demo UAS

*Print kartu ini dan bawa saat presentasi sebagai panduan cepat*

---

## 📋 AGENDA (20 menit)
1. Opening (2 min)
2. Background (2 min)
3. **DEMO** (8-10 min) ⭐
4. Technical (3-4 min)
5. Closing (2 min)
6. Q&A (3-5 min)

---

## 🎯 KEY POINTS

### Masalah:
- ❌ Sulit hitung kalori manual
- ❌ Info nutrisi terbatas
- ❌ Tracking memakan waktu

### Solusi:
- ✅ AI deteksi otomatis
- ✅ Database 18 makanan
- ✅ UI modern & mudah
- ✅ Kalori real-time

---

## 🚀 DEMO FLOW

1. **Home Page** → Jelaskan UI
2. **Pilih Model** → MobileNetV2
3. **Upload Foto** → Rendang
4. **Loading** → Explain stages
5. **Hasil** → 290 kal, 85% confidence
6. **Recalculate** → 150g → 200g
7. **History** → Show saved items
8. **Upload 2** → Nasi Goreng
9. **(Optional)** → Try different model

---

## 💻 TECH STACK

**Backend:**
- Python + Flask
- TensorFlow/Keras
- NumPy

**Frontend:**
- HTML5
- Tailwind CSS
- JavaScript

**ML:**
- MobileNetV2
- EfficientNetB0
- Custom CNN

---

## 📊 QUICK STATS

### Models:
```
MobileNetV2:   85% | 2-3s | 14MB
EfficientNet:  90% | 3-4s | 29MB
SimpleCNN:     75% | 1-2s | 5MB
```

### Dataset:
```
Classes:    18 makanan
Images:     ~10,000+
Per class:  ~500-1000
Split:      80/20
```

### Top Calories (per 100g):
```
Dendeng:    300 kal
Ayam Goreng: 275 kal
Nasi Goreng: 267 kal
Rendang:    193 kal
```

---

## 🎤 OPENING SCRIPT

> "Assalamualaikum/Selamat pagi"
>
> "Nama saya [NAMA], NIM [NIM]"
>
> "Saya akan presentasikan:"
> **"CALORIE COUNTER AI"**
> "Aplikasi Penghitung Kalori Makanan
> Indonesia Berbasis AI"

---

## 🎬 DEMO SCRIPT

### 1. Home Page
> "Ini halaman utama dengan:
> - Navigation bar
> - Model selection
> - Upload area
> - Features section"

### 2. Model Selection
> "3 pilihan model:
> - MobileNetV2: Cepat & efisien
> - EfficientNet: Akurasi tinggi
> - SimpleCNN: Model dasar
>
> Saya pilih MobileNetV2"

### 3. Upload
> "Saya upload foto Rendang...
> Validasi: JPG/PNG/WEBP, max 16MB"

### 4. Loading
> "Proses analisis:
> - Validasi gambar
> - Preprocessing
> - AI analysis
> - Hitung kalori
> Hanya 2-3 detik"

### 5. Result
> "Hasil: RENDANG
> - Kalori: 290 kal (150g)
> - Per 100g: 193 kal
> - Confidence: [X]%
> - Model: MobileNetV2"

### 6. Recalculate
> "Fitur unggulan: dynamic calculation
> Ubah 150g → 200g
> Kalori update otomatis"

### 7. History
> "Auto-save ke riwayat
> Bisa lihat, edit, hapus"

---

## 🔧 TECHNICAL POINTS

### Arsitektur:
```
User → Flask → Preprocessing
     → Model → Prediction
     → Calorie DB → Response
```

### Preprocessing:
```python
# Resize
img = load_img(path, (128,128))

# Normalize
img_array = img / 255.0

# Predict
preds = model.predict(img_array)
```

### Features:
- Model caching
- Data augmentation
- Transfer learning
- Fine-tuning

---

## ✅ KELEBIHAN

- 🎨 UI modern & attractive
- ⚡ Cepat (2-3 detik)
- 🎯 Akurat (~85%)
- 🔧 Fleksibel (3 models)
- 📱 Responsive design
- 💾 Auto-save history

---

## ⚠️ KEKURANGAN

- 18 makanan saja
- Perlu internet
- Porsi manual
- Hanya kalori

### Solusi:
- Tambah dataset
- Mobile app (offline)
- AI portion estimation
- Full nutrition info

---

## 🎓 CLOSING SCRIPT

> "**Kesimpulan:**
>
> ✅ Berhasil implementasi Deep Learning
> untuk klasifikasi makanan Indonesia
>
> ✅ Akurasi ~85% untuk 18 makanan
>
> ✅ UI modern dan user-friendly
>
> ✅ Sistem scalable untuk development
>
> **Terima kasih!**
> Siap menjawab pertanyaan"

---

## ❓ Q&A QUICK ANSWERS

**Q: Akurasi model?**
A: MobileNet ~85%, EfficientNet ~90%

**Q: Kenapa 18 makanan?**
A: Keterbatasan waktu & resource.
   Fokus pada makanan populer.
   Plan: tambah 20+ makanan.

**Q: Sumber data kalori?**
A: Kemenkes RI, USDA Database,
   penelitian ilmiah, verified.

**Q: Bisa mobile app?**
A: Ya! Plan: PWA atau React Native
   dengan TensorFlow Lite (offline)

**Q: Waktu training?**
A: MobileNet: 2-3 jam
   EfficientNet: 4-6 jam
   (dengan GPU)

**Q: Makanan mirip?**
A: Model belajar dari tekstur, warna,
   bentuk. Ada confidence score
   untuk indikasi keyakinan.

**Q: Sudah deploy?**
A: Masih localhost. Siap deploy
   ke Heroku/GCP/Netlify.

**Q: Jika salah deteksi?**
A: Future: manual correction feature.
   User bisa pilih makanan yang benar.

---

## 🚨 EMERGENCY NOTES

### App Crash:
1. Restart cepat
2. Explain ke audience
3. Use screenshots/video backup
4. Continue with slides

### Lupa Script:
1. Napas dalam
2. Lihat slide
3. Explain dengan bahasa sendiri
4. Fokus poin utama

### Pertanyaan Sulit:
1. "Pertanyaan bagus"
2. Jika tidak tahu: jujur
3. Offer follow-up
4. Jangan mengada-ada

---

## 💪 CONFIDENCE BOOSTERS

✅ "Saya sudah persiapan baik"
✅ "Saya paham project saya"
✅ "Kesalahan kecil itu wajar"
✅ "Audience ingin saya sukses"

**BREATHE. SMILE. YOU GOT THIS!**

---

## 🎯 FINAL REMINDERS

- [ ] Speak clearly & slowly
- [ ] Eye contact
- [ ] Smile & enthusiasm
- [ ] Pause after key points
- [ ] Don't read slides
- [ ] Stay calm if error
- [ ] Enjoy the moment!

---

## 📞 EMERGENCY

Dosen: [NOMOR]
Teman: [NOMOR]
IT Support: [NOMOR]

---

## ⏱️ TIME CHECK

| Time | Section |
|------|---------|
| 0:00 | Opening |
| 2:00 | Background |
| 4:00 | **START DEMO** |
| 14:00 | Technical |
| 17:00 | Closing |
| 19:00 | Q&A |

**STAY ON TRACK!**

---

## 🎬 DEMO COMMANDS

```bash
# Start app
python app.py

# Open browser
http://localhost:5000

# Demo images location
demo_images/
```

---

## 📊 METRICS CHEAT SHEET

**Accuracy:**
- Training: ~88%
- Validation: ~85%
- Precision: ~84%
- Recall: ~83%

**Dataset:**
- Total: 10,000+ images
- Classes: 18
- Augmentation: Yes

**Performance:**
- Inference: 2-3 seconds
- Model size: 14-29 MB
- Input: 128x128 or 224x224

---

## 🌟 HIGHLIGHT FEATURES

1. **3 AI Models** - Flexibility
2. **Dynamic Portion** - Accuracy
3. **Auto History** - Convenience
4. **Modern UI** - User Experience
5. **Fast Analysis** - Efficiency

---

## 🎯 KEY TAKEAWAYS

> "AI dapat dimanfaatkan untuk
> mendukung kesehatan masyarakat
> dengan cara yang praktis dan
> accessible"

> "Transfer learning sangat efektif
> untuk dataset terbatas"

> "Balance antara akurasi dan
> kecepatan penting untuk UX"

---

## 🏆 SUCCESS CRITERIA

✅ Demo berjalan lancar
✅ Explain clearly
✅ Answer questions confidently
✅ Stay within time
✅ Show enthusiasm
✅ Professional demeanor

---

*Print kartu ini 2-sided atau sebagai booklet kecil*
*Bawa sebagai quick reference saat presentasi*

**GOOD LUCK! 🚀**

---

**Last Check Before Start:**
- [ ] Laptop charged
- [ ] App tested
- [ ] Slides ready
- [ ] Demo images ready
- [ ] Water bottle
- [ ] Deep breath
- [ ] SMILE! 😊

**YOU'RE READY TO ROCK THIS!** 🎸🔥
