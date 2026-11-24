# Arsitektur Arun-D

## Dua dunia
- **ML (offline):** dataset → preprocessing → augmentasi → CNN → loss → backprop → optimizer →
  evaluasi → artefak (`model.pt`, `classes.json`, `metrics.json`).
- **Aplikasi (online):** React → FastAPI → Predictor (memuat `model.pt` sekali) → MySQL.

## Alur satu prediksi
1. React mengirim `POST /api/predict` (multipart).
2. FastAPI memvalidasi tipe dan ukuran file.
3. OpenCV men-decode bytes menjadi gambar RGB.
4. Transform evaluasi (resize 32×32, normalisasi) **identik dengan validation/test**.
5. Model memprediksi gambar asli dan flip horizontal; probabilitas keduanya dirata-ratakan sebelum top-3 dipilih.
6. Hasil disimpan ke tabel `predictions`, lalu dikembalikan sebagai JSON.

## Keputusan desain
| Topik | Pilihan | Alasan |
|---|---|---|
| Dataset | CIFAR-10 | Kecil, legal untuk edukasi, cepat dilatih |
| Model | CNN VGG-mini + BatchNorm + Dropout (~1,2 juta parameter) | Dari nol, muat di RTX 3050 |
| Loss | CrossEntropyLoss | Standar klasifikasi multi-kelas |
| Optimizer | AdamW (opsional SGD+Nesterov) | Konvergen cepat, mudah dibandingkan |
| Presisi | Mixed precision (AMP) | Hemat VRAM, lebih cepat |
| Split data | 45.000 train / 5.000 val / 10.000 test | Test tidak disentuh saat tuning |
| Model di backend | Dimuat sekali (singleton) | Muat model per request sangat lambat |
| Gambar upload | Tidak disimpan, hanya metadata | Hemat ruang, privasi |
