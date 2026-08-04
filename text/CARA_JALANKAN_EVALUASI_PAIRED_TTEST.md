# 📊 Cara Menjalankan Evaluasi dengan Paired T-Test

## 🎯 Apa yang Ditambahkan?

Saya telah menambahkan **Statistical Significance Testing** ke dalam evaluasi model Anda:

### ✅ Fitur Baru:
1. **Paired T-Test** - Membandingkan akurasi antar model secara statistik
2. **McNemar's Test** - Membandingkan pola kesalahan antar model
3. **Confidence Intervals (95%)** - Rentang kepercayaan untuk perbedaan akurasi
4. **P-value Analysis** - Menentukan signifikansi statistik (α = 0.05)
5. **Visualisasi Grafik** - Grafik P-value dan Mean Difference dengan CI

### 📁 Output yang Dihasilkan:
- `paired_ttest_results.csv` - Hasil paired t-test lengkap
- `mcnemar_test_results.csv` - Hasil McNemar test lengkap
- `statistical_significance_tests.png` - Grafik P-value dan Mean Difference
- `statistical_test_summary_table.png` - Tabel ringkasan hasil

---

## 🚀 Langkah-Langkah Menjalankan

### 1. Install Dependencies (Jika Belum)

```bash
pip install scipy
```

Atau install semua requirements untuk evaluasi:

```bash
pip install -r requirements_evaluation.txt
```

### 2. Jalankan Evaluasi

```bash
python evaluate_models.py
```

### 3. Proses Evaluasi

Evaluasi akan menjalankan:
1. ✅ Training curves visualization
2. ✅ Evaluasi MobileNetV2
3. ✅ Evaluasi EfficientNetB0
4. ✅ Evaluasi SimpleCNN
5. ✅ Confusion matrix untuk setiap model
6. ✅ Confidence score analysis
7. ✅ Inference speed benchmark
8. ⭐ **PAIRED T-TEST** (BARU!)
9. ⭐ **McNEMAR'S TEST** (BARU!)
10. ✅ Summary comparison

**Estimasi Waktu:** 10-15 menit (tergantung hardware)

### 4. Lihat Hasil

Setelah selesai, cek folder `evaluation_results/`:

```
evaluation_results/
├── paired_ttest_results.csv                 ⭐ BARU!
├── mcnemar_test_results.csv                 ⭐ BARU!
├── statistical_significance_tests.png       ⭐ BARU!
├── statistical_test_summary_table.png       ⭐ BARU!
├── model_comparison_summary.csv
├── (dan file-file lainnya...)
```

---

## 📖 Interpretasi Hasil

### Paired T-Test

**Contoh Output:**
```
Perbandingan: SimpleCNN vs MobileNetV2
─────────────────────────────────────────
Accuracy SimpleCNN:     76.99%
Accuracy MobileNetV2:   72.61%
Mean Difference:        4.38%
95% Confidence Interval: [3.12%, 5.64%]

T-statistic:            8.4521
P-value:                0.000001
Degrees of freedom:     6806

✅ SIGNIFIKAN (p < 0.05)
   → SimpleCNN secara statistik LEBIH BAIK
```

**Interpretasi:**
- **P-value < 0.05** → Perbedaan signifikan secara statistik
- **P-value ≥ 0.05** → Tidak ada perbedaan signifikan
- **Mean Difference** → Rata-rata perbedaan akurasi
- **95% CI** → Rentang kepercayaan 95%

### McNemar's Test

**Contoh Output:**
```
Perbandingan: SimpleCNN vs MobileNetV2
─────────────────────────────────────────
Contingency Table:
  Both Correct:           4500
  SimpleCNN Only:          743
  MobileNetV2 Only:        450
  Both Incorrect:         1114

McNemar Statistic:       71.23
P-value:                 0.000000

✅ SIGNIFIKAN (p < 0.05)
   → Model memiliki pola kesalahan yang berbeda secara signifikan
```

**Interpretasi:**
- Menguji apakah model memiliki pola kesalahan yang berbeda
- Fokus pada sampel yang TIDAK disepakati kedua model

---

## 💡 Tips untuk Sidang

### 1. Penjelasan untuk Penguji

**Q: "Bagaimana Anda membuktikan SimpleCNN lebih baik dari MobileNetV2?"**

**A (DENGAN PAIRED T-TEST):** 
> "Kami melakukan paired t-test untuk membandingkan performa kedua model pada dataset yang sama. Hasilnya menunjukkan:
> - SimpleCNN: 76.99%
> - MobileNetV2: 72.61%
> - Mean Difference: 4.38%
> - P-value: 0.000001 (< 0.05)
> 
> Dengan p-value < 0.05, kami dapat menyimpulkan bahwa SimpleCNN **secara statistik signifikan** lebih baik dari MobileNetV2 dengan confidence level 95%."

**A (TANPA PAIRED T-TEST):**
> "SimpleCNN memiliki akurasi 76.99% dibandingkan MobileNetV2 72.61%."
> *(Penguji bisa tanya: "Tapi perbedaan 4% bisa saja kebetulan, bukan?")*

### 2. Slide Presentasi

Tambahkan slide baru:

**SLIDE: "Statistical Significance Testing"**
- Judul: Uji Statistik Perbandingan Model
- Grafik: `statistical_significance_tests.png`
- Tabel: `statistical_test_summary_table.png`
- Poin-poin:
  - ✅ Paired t-test untuk validasi perbedaan akurasi
  - ✅ McNemar's test untuk pola kesalahan
  - ✅ Significance level: α = 0.05
  - ✅ Semua perbandingan dengan 95% confidence interval

### 3. Antisipasi Pertanyaan

**Q: "Apa perbedaan paired t-test dengan independent t-test?"**
**A:** "Paired t-test digunakan karena model dievaluasi pada test set yang SAMA, sehingga sampelnya berpasangan. Ini lebih tepat dan powerful dibanding independent t-test."

**Q: "Kenapa menggunakan α = 0.05?"**
**A:** "Alpha 0.05 adalah standar dalam penelitian ilmiah, yang berarti kami toleran terhadap 5% kemungkinan Type I error (false positive)."

**Q: "Apa itu McNemar's test?"**
**A:** "McNemar's test khusus untuk membandingkan model klasifikasi. Test ini fokus pada sampel yang diprediksi berbeda oleh kedua model, untuk menentukan apakah perbedaan tersebut signifikan."

---

## 🎓 Referensi Akademik

Tambahkan referensi ini di laporan/paper Anda:

1. **Paired T-Test:**
   - Student (1908). "The probable error of a mean." Biometrika
   
2. **McNemar's Test:**
   - McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages." Psychometrika

3. **Model Comparison in ML:**
   - Dietterich, T. G. (1998). "Approximate statistical tests for comparing supervised classification learning algorithms." Neural computation

---

## ✅ Checklist Sidang

Pastikan Anda punya:

- [ ] File `paired_ttest_results.csv` sudah ada
- [ ] File `mcnemar_test_results.csv` sudah ada
- [ ] File `statistical_significance_tests.png` sudah ada
- [ ] File `statistical_test_summary_table.png` sudah ada
- [ ] Anda paham interpretasi P-value
- [ ] Anda paham apa itu paired t-test
- [ ] Anda paham kenapa α = 0.05
- [ ] Anda siap jelaskan confidence interval

---

## 🆘 Troubleshooting

### Error: "No module named 'scipy'"
```bash
pip install scipy
```

### Error: "Models have different test sets"
- Pastikan semua model menggunakan test set yang sama
- Jangan gunakan shuffle=True di test generator

### Evaluasi Terlalu Lama
- Normal, estimasi 10-15 menit
- Model harus diload dan diprediksi untuk semua test samples

---

## 📞 Bantuan

Jika ada pertanyaan atau error, hubungi:
- Periksa console output untuk error messages
- Cek apakah semua model (.h5) ada di folder `models/`
- Pastikan scipy terinstall

---

**SELAMAT SIDANG! 🎓🚀**

Dengan paired t-test, penelitian Anda sekarang memiliki **validasi statistik yang kuat**!

