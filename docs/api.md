# API

| Method | Endpoint | Fungsi |
|---|---|---|
| POST | `/api/predict` | Upload gambar (field `file`), kembalikan prediksi |
| GET | `/api/history?limit=&offset=` | Riwayat prediksi (terbaru dulu) |
| DELETE | `/api/history/{id}` | Hapus satu riwayat |
| GET | `/api/stats` | Total, rata-rata confidence & latensi, per kelas, 14 hari terakhir |
| GET | `/api/model/info` | Versi, akurasi test, jumlah parameter, kelas |
| GET | `/api/model/metrics` | Kurva loss/akurasi, metrik per kelas, confusion matrix |
| GET | `/api/health` | Cek server |

## Contoh `/api/predict`
```json
{
  "id": 12,
  "predicted": "cat",
  "confidence": 0.9132,
  "top_k": [{"label": "cat", "score": 0.9132}, {"label": "dog", "score": 0.0621}, {"label": "deer", "score": 0.0114}],
  "inference_ms": 14.2,
  "model_version": "aruna-cnn-v1"
}
```

## Kode error
| Kode | Arti |
|---|---|
| 400 | File kosong atau bukan gambar valid |
| 413 | File melebihi batas (default 5 MB) |
| 415 | Format bukan JPG/PNG/WEBP/BMP |
| 503 | Model belum dilatih |
