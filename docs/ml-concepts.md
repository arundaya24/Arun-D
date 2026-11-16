# Konsep Inti: bagaimana model belajar

## 1. Forward propagation
Gambar 3×32×32 melewati lapisan berurutan. Tiap lapisan adalah fungsi: `output = f(input; bobot)`.

- **Conv2d 3×3:** filter kecil digeser di atas gambar; tiap posisi menghitung jumlah perkalian
  piksel × bobot. Filter awal belajar mendeteksi tepi, lapisan dalam mendeteksi bentuk dan bagian objek.
- **BatchNorm:** menormalkan aktivasi tiap batch (rata-rata 0, varians 1) lalu menskalakan ulang
  dengan parameter yang dipelajari. Membuat training lebih stabil dan cepat.
- **ReLU:** `max(0, x)`. Tanpa non-linearitas, tumpukan lapisan hanya setara satu lapisan linear.
- **MaxPool:** mengambil nilai terbesar tiap blok 2×2, memperkecil ukuran (32→16→8→4) dan memberi
  sedikit toleransi terhadap pergeseran posisi.
- **Global Average Pooling + Linear:** 256 peta fitur dirata-rata menjadi 256 angka, lalu
  dipetakan ke 10 skor (**logits**), satu per kelas.

## 2. Loss function
Logits diubah menjadi probabilitas dengan **softmax**: `p_i = exp(z_i) / Σ exp(z_j)`.
**Cross-Entropy** mengukur seberapa kecil probabilitas yang diberikan ke kelas yang benar:

`L = -log(p_benar)`

Jika model yakin dan benar (p=0,95) → L≈0,05. Jika yakin tapi salah (p benar=0,01) → L≈4,6.
Itulah mengapa loss menghukum keras kesalahan yang percaya diri. Di PyTorch, `CrossEntropyLoss`
menggabungkan softmax + log + pengambilan nilai, sehingga model cukup mengeluarkan logits.

## 3. Backpropagation
Tujuan: tahu seberapa besar tiap bobot `w` berkontribusi pada loss, yaitu `∂L/∂w`.
Dengan **aturan rantai**, gradien dihitung mundur dari loss ke lapisan pertama:

`∂L/∂w₁ = ∂L/∂out · ∂out/∂h · ∂h/∂w₁`

PyTorch merekam semua operasi forward dalam *computational graph*; `loss.backward()` menelusurinya
mundur dan mengisi `param.grad`. Lihat `ml/scripts/gradient_demo.py` untuk versi manual 1 neuron
yang diverifikasi terhadap autograd.

## 4. Gradient descent & optimizer
Gradien menunjuk arah kenaikan loss tercepat, jadi bobot digeser berlawanan arah:

`w ← w - lr · ∂L/∂w`

- **lr (learning rate)** terlalu besar → loss melompat/divergen; terlalu kecil → belajar lambat.
- **Mini-batch:** gradien dihitung dari 128 gambar sekaligus (cukup akurat, jauh lebih murah
  daripada seluruh dataset).
- **SGD + momentum:** menambah "kelembaman" agar gerakan lebih mulus.
- **Adam/AdamW:** menyesuaikan langkah tiap parameter berdasarkan rata-rata gradien (momentum) dan
  rata-rata kuadrat gradien (skala). AdamW memisahkan *weight decay* dengan benar.
- **Cosine scheduler:** lr turun perlahan ke mendekati nol agar model "mengendap" di minimum yang baik.
- **Gradient clipping:** membatasi norma gradien untuk mencegah ledakan.

## 5. Siklus training (`train.py`)
```
untuk tiap epoch:
    untuk tiap batch (x, y):
        zero_grad()          # buang gradien batch sebelumnya
        logits = model(x)    # forward
        loss = CE(logits,y)  # loss
        loss.backward()      # backprop
        optimizer.step()     # update bobot
    evaluasi di validation (model.eval(), tanpa gradien)
    simpan model jika val_acc terbaik
```

## 6. Overfitting dan cara melawannya
Tanda: akurasi train terus naik, akurasi validasi mandek/turun (celah kedua kurva melebar).
Yang dipakai di proyek ini: **data augmentation** (crop acak, flip, color jitter), **Dropout2d**,
**weight decay**, **early stopping**, dan **pemilihan model berdasarkan validasi**.

## 7. Mengapa tiga split
- **Train:** untuk memperbarui bobot.
- **Validation:** untuk memilih epoch/hyperparameter. Model "mengintip" ini secara tidak langsung.
- **Test:** dipakai SEKALI di akhir untuk estimasi kinerja jujur pada data baru.

## Eksperimen yang disarankan
1. Ganti `optimizer: sgd` dan `lr: 0.05`, bandingkan kurva dengan Adam.
2. Matikan augmentasi di `transforms.py`, amati overfitting.
3. Set `dropout: 0.0` lalu `0.5`, bandingkan celah train-val.
4. Ubah `channels` menjadi `[32, 64, 128]` dan lihat trade-off akurasi vs kecepatan.
5. Naikkan `lr` ke `0.1` pada Adam dan amati loss divergen.
