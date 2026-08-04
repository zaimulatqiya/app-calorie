# 📸 PANDUAN FOTO DEMO
## Persiapan Gambar Makanan untuk Demo

---

## 🎯 TUJUAN

Menyiapkan foto-foto makanan berkualitas baik untuk demo aplikasi saat presentasi UAS.

**Foto yang baik = Demo yang smooth = Presentasi yang impressive!**

---

## 📋 DAFTAR FOTO YANG DIBUTUHKAN

### **MINIMUM (3-4 foto):**
```
1. ✅ Rendang          (High confidence)
2. ✅ Nasi Goreng     (Popular food)
3. ✅ Sate            (Distinctive)
4. ⚠️ Low quality     (Untuk demo limitasi)
```

### **IDEAL (5-7 foto):**
```
1. ✅ Rendang
2. ✅ Nasi Goreng
3. ✅ Sate
4. ✅ Gudeg
5. ✅ Ayam Goreng
6. ✅ Bakso
7. ⚠️ Low quality (1-2 foto)
```

---

## 🔍 KRITERIA FOTO YANG BAIK

### ✅ **GOOD PHOTO:**

**Kualitas:**
- Resolusi minimal: 640x640 pixels
- Format: JPG, PNG, atau WEBP
- File size: 100KB - 5MB
- Tidak blur atau pixelated

**Komposisi:**
- Makanan terlihat jelas
- Angle dari atas (top-down) atau 45°
- Background simple, tidak ramai
- Lighting baik (tidak terlalu gelap/terang)

**Konten:**
- Fokus pada satu jenis makanan
- Porsi terlihat jelas
- Warna natural (tidak over-saturated)
- Tidak ada watermark besar

---

### ❌ **BAD PHOTO:**

**Hindari:**
- Blur atau out of focus
- Terlalu gelap atau terlalu terang
- Angle yang aneh (terlalu miring)
- Background yang ramai/distracting
- Makanan tercampur banyak jenis
- Watermark besar menutupi makanan
- File corrupt atau rusak

---

## 📁 STRUKTUR FOLDER

```
📁 demo_images/
├── 📸 01_rendang.jpg
├── 📸 02_nasi_goreng.jpg
├── 📸 03_sate.jpg
├── 📸 04_gudeg.jpg
├── 📸 05_ayam_goreng.jpg
├── 📸 06_bakso.jpg (optional)
└── 📸 07_low_quality.jpg (untuk demo limitasi)
```

**Naming Convention:**
- Lowercase
- Underscore untuk spasi
- Nomor urut di depan
- Deskriptif

---

## 🌐 SUMBER FOTO

### **Option 1: Foto Sendiri (BEST)**
✅ Original & authentic
✅ Tidak ada copyright issue
✅ Bisa disesuaikan dengan kebutuhan

**Tips:**
- Gunakan smartphone dengan kamera bagus
- Natural lighting (dekat jendela)
- Angle 45° atau top-down
- Edit sedikit jika perlu (brightness, contrast)

---

### **Option 2: Free Stock Photos**

**Recommended Sites:**

1. **Unsplash** (unsplash.com)
   - Free high-quality images
   - No attribution required
   - Search: "indonesian food"

2. **Pexels** (pexels.com)
   - Free stock photos
   - Good quality
   - Search: "rendang", "nasi goreng", etc.

3. **Pixabay** (pixabay.com)
   - Free images
   - No copyright
   - Various quality

4. **Google Images**
   - Filter: "Creative Commons licenses"
   - Check usage rights
   - Verify quality

**Search Keywords:**
```
- "rendang"
- "nasi goreng"
- "sate ayam"
- "gudeg jogja"
- "ayam goreng"
- "bakso"
- "indonesian food"
- "traditional indonesian cuisine"
```

---

### **Option 3: Dataset Training**

Jika Anda punya akses ke dataset training:
- Pilih gambar terbaik dari validation set
- Pastikan tidak over-represented
- Variasi angle dan lighting

---

## 🎨 EDITING FOTO (Optional)

### **Basic Editing:**

**Tools:**
- **Windows Photos** (built-in)
- **Paint.NET** (free)
- **GIMP** (free, advanced)
- **Canva** (online, free tier)

**Adjustments:**
```
✅ Brightness: +10 to +20
✅ Contrast: +5 to +15
✅ Saturation: -5 to +10 (jangan berlebihan)
✅ Crop: Remove distractions
✅ Resize: 1024x1024 atau 800x800
```

**Don't Overdo:**
❌ Jangan terlalu saturated (warna tidak natural)
❌ Jangan terlalu sharp (terlihat fake)
❌ Jangan tambah filter berlebihan

---

## 📏 SPESIFIKASI TEKNIS

### **Recommended:**
```
Format:      JPG atau PNG
Resolution:  800x800 to 1920x1920
Aspect:      Square (1:1) atau landscape (4:3)
File Size:   200KB - 2MB
Color:       RGB
Quality:     80-90% (JPG)
```

### **Minimum:**
```
Format:      JPG, PNG, WEBP
Resolution:  640x640 minimum
File Size:   100KB - 5MB
```

---

## ✅ CHECKLIST FOTO

**Untuk Setiap Foto:**

- [ ] Resolusi cukup (min 640x640)
- [ ] Format supported (JPG/PNG/WEBP)
- [ ] File size reasonable (100KB-5MB)
- [ ] Makanan terlihat jelas
- [ ] Lighting baik
- [ ] Background simple
- [ ] Tidak blur
- [ ] Nama file deskriptif
- [ ] Saved di folder `demo_images/`

---

## 🎯 STRATEGI DEMO

### **Foto 1: Rendang (High Confidence)**
**Tujuan:** Tunjukkan akurasi model
**Kriteria:**
- Foto berkualitas tinggi
- Rendang terlihat jelas
- Angle bagus
- Expected confidence: >85%

---

### **Foto 2: Nasi Goreng (Popular)**
**Tujuan:** Tunjukkan makanan populer
**Kriteria:**
- Nasi goreng khas Indonesia
- Dengan telur/ayam (typical)
- Warna appetizing
- Expected confidence: >80%

---

### **Foto 3: Sate (Distinctive)**
**Tujuan:** Tunjukkan makanan dengan bentuk unik
**Kriteria:**
- Sate di tusuk sate
- Terlihat jelas
- Dengan bumbu kacang (optional)
- Expected confidence: >85%

---

### **Foto 4-6: Variasi (Optional)**
**Tujuan:** Tunjukkan kemampuan model
**Kriteria:**
- Berbeda dari 3 pertama
- Kualitas baik
- Representatif

---

### **Foto Low Quality (Demo Limitasi)**
**Tujuan:** Tunjukkan limitation model
**Kriteria:**
- Blur atau gelap
- Angle buruk
- Expected: Lower confidence atau salah deteksi

**Gunakan untuk:**
- Explain model limitations
- Show importance of good photo
- Demonstrate confidence score

---

## 🔄 BACKUP PLAN

### **Jika Foto Tidak Tersedia:**

**Plan A:**
- Gunakan foto dari internet (free stock)
- Pastikan quality baik

**Plan B:**
- Foto makanan di rumah/warung
- Edit sedikit jika perlu

**Plan C:**
- Gunakan foto dari dataset training
- Pilih yang terbaik

**Plan D:**
- Screenshot dari Google Images
- Crop dan resize

---

## 📸 TIPS FOTOGRAFI MAKANAN

### **Lighting:**
```
✅ Natural light (dekat jendela)
✅ Diffused light (tidak harsh)
✅ Avoid direct sunlight
✅ Avoid flash (jika bisa)
```

### **Angle:**
```
✅ Top-down (90°) - Best for flat foods
✅ 45° angle - Best for layered foods
✅ Eye level - For tall foods
```

### **Composition:**
```
✅ Rule of thirds
✅ Negative space
✅ Simple background
✅ Focus on food
```

### **Styling (Optional):**
```
✅ Garnish (daun, sambal)
✅ Props minimal (sendok, piring)
✅ Clean presentation
```

---

## 🎬 SAAT DEMO

### **Persiapan:**
```
1. Buka folder demo_images/
2. Sort by name (01, 02, 03...)
3. Preview semua foto
4. Pastikan semua bisa dibuka
```

### **Urutan Upload:**
```
1. Rendang (impressive start)
2. Nasi Goreng (popular food)
3. Sate (distinctive)
4. (Optional) Makanan lain
5. (Optional) Low quality untuk demo limitasi
```

### **Narasi:**
```
"Saya akan upload foto Rendang..."
[Upload]
"Perhatikan bahwa foto ini berkualitas baik
dengan lighting yang cukup..."
[Tunggu hasil]
"Dan hasilnya, model mendeteksi dengan
confidence [X]%..."
```

---

## 🚨 TROUBLESHOOTING

### **Problem: Foto terlalu besar**
**Solution:**
- Resize dengan Paint/Photos
- Compress dengan online tool
- Target: <2MB

### **Problem: Foto blur**
**Solution:**
- Cari foto lain yang lebih sharp
- Atau gunakan untuk demo limitasi

### **Problem: Model salah deteksi**
**Solution:**
- Coba foto lain dari makanan yang sama
- Atau explain sebagai limitation
- Tunjukkan confidence score rendah

### **Problem: Confidence rendah**
**Solution:**
- Normal jika foto kurang bagus
- Explain faktor yang mempengaruhi
- Tunjukkan dengan foto lebih baik

---

## 📊 EXPECTED RESULTS

### **Good Photo:**
```
Confidence: 80-95%
Detection: Correct
Time: 2-3 seconds
```

### **Average Photo:**
```
Confidence: 60-80%
Detection: Usually correct
Time: 2-3 seconds
```

### **Poor Photo:**
```
Confidence: <60%
Detection: May be incorrect
Time: 2-3 seconds
```

---

## ✅ FINAL CHECKLIST

**Before Presentation:**

- [ ] 3-7 foto sudah disiapkan
- [ ] Semua foto di folder `demo_images/`
- [ ] Nama file deskriptif dan terurut
- [ ] Semua foto bisa dibuka
- [ ] Tested dengan aplikasi
- [ ] Confidence level acceptable
- [ ] Backup foto di USB/cloud

**Day of Presentation:**

- [ ] Folder demo_images accessible
- [ ] Foto sorted by name
- [ ] Preview sekali lagi
- [ ] Ready to upload

---

## 💡 PRO TIPS

1. **Test Dulu**
   - Upload semua foto ke aplikasi
   - Check confidence level
   - Pilih yang terbaik

2. **Backup**
   - Siapkan 2x lebih banyak foto
   - Jika satu gagal, ada alternatif

3. **Variety**
   - Berbeda jenis makanan
   - Berbeda angle
   - Berbeda lighting

4. **Quality Over Quantity**
   - 3 foto bagus > 10 foto biasa
   - Focus on best results

5. **Story**
   - Pilih foto yang bisa diceritakan
   - "Ini foto rendang dari..."

---

## 🎯 SUCCESS CRITERIA

**Foto demo berhasil jika:**

✅ Model deteksi dengan benar
✅ Confidence level tinggi (>80%)
✅ Foto terlihat professional
✅ Audience impressed
✅ Demo berjalan smooth

---

## 📁 DOWNLOAD LINKS

**Free Stock Photo Sites:**
- Unsplash: https://unsplash.com
- Pexels: https://pexels.com
- Pixabay: https://pixabay.com

**Image Editing:**
- Paint.NET: https://www.getpaint.net/
- GIMP: https://www.gimp.org/
- Canva: https://www.canva.com/

**Image Compression:**
- TinyPNG: https://tinypng.com/
- Compressor.io: https://compressor.io/

---

## 🌟 FINAL WORDS

**Good photos = Good demo = Good presentation!**

Jangan underestimate pentingnya foto yang baik.

**Invest waktu untuk:**
- Cari/buat foto berkualitas
- Test dengan aplikasi
- Pilih yang terbaik

**It's worth it!** 💪

---

**GOOD LUCK!** 📸✨

**Happy Photo Hunting!** 🎯

---

*Panduan ini membantu Anda menyiapkan foto demo yang berkualitas*
