# Dashboard Klasterisasi Risiko DBD Jawa Barat 2025 — Panduan Run & Deploy

Kelompok 5 · Program Studi Komputasi Statistik (D-IV), Peminatan Sains Data · Politeknik Statistika STIS.

**Logo institusi.** Dashboard memakai lambang sementara `assets/logo_stis.svg`. Untuk memakai logo resmi,
simpan berkasnya sebagai **`assets/logo_stis.png`** (PNG latar transparan, minimal 200×200 px);
dashboard otomatis memakainya pada header setiap halaman, footer, dan favicon.

**Mode tampilan.** Tersedia sakelar *Mode gelap* di bagian bawah sidebar.

## A. Menyiapkan data
1. Jalankan notebook **Klasterisasi_DBD_Jabar_2025_v3.ipynb** sampai selesai (Run All).
2. Buka folder `output_analisis` hasil notebook, lalu salin dua berkas ini ke folder `data/`:
   - `Hasil_Klasterisasi_DBD_Jabar_2025.xlsx`
   - `klaster_jabar_2025.geojson`
3. `Master_Dataset_DBD_Jabar_2025.xlsx` sudah ada di `data/`.

Isi folder yang benar:
```
dashboard_dbd/
├── app.py
├── requirements.txt
└── data/
    ├── Master_Dataset_DBD_Jabar_2025.xlsx
    ├── Hasil_Klasterisasi_DBD_Jabar_2025.xlsx
    └── klaster_jabar_2025.geojson
```

## B. Menjalankan di laptop (Windows)
1. Buka **Anaconda Prompt** atau **Command Prompt**.
2. Masuk ke folder dashboard, misalnya:
   ```
   cd "E:\FILE TINGKAT 3\SEMESTER 6\Data Mining (Datmin)\PROJECT KELOMPOK DATMIN\dashboard_dbd"
   ```
3. Pasang paket (cukup sekali):
   ```
   pip install -r requirements.txt
   ```
4. Jalankan:
   ```
   streamlit run app.py
   ```
5. Browser terbuka otomatis di `http://localhost:8501`. Tekan `Ctrl + C` di Command Prompt untuk berhenti.

## C. Deploy gratis ke Streamlit Community Cloud
1. Buat akun di https://github.com, lalu buat repository baru (misalnya `dashboard-dbd-jabar`) bertipe **Public**.
2. Klik **Add file → Upload files**, lalu seret `app.py`, `requirements.txt`, dan folder `data/` (beserta ketiga berkasnya). Klik **Commit changes**.
3. Buka https://share.streamlit.io dan login dengan akun GitHub.
4. Klik **Create app → Deploy a public app from GitHub**. Pilih repository, branch `main`, main file path `app.py`. Pada *App URL*, isi nama singkat (misalnya `dbd-jabar-2025`).
5. Klik **Deploy**. Tunggu 2–5 menit hingga aplikasi tampil. Tautan berbentuk `https://dbd-jabar-2025.streamlit.app`; cantumkan di laporan (subbab 4.8) dan di slide 26 PPT.
6. Jika ada perubahan, cukup upload ulang berkas di GitHub; aplikasi otomatis diperbarui.

Masalah umum:
- Muncul "File hasil notebook belum ditemukan" → berkas Excel hasil belum ada di `data/`, atau nama berkasnya berbeda.
- Peta berupa titik, bukan poligon → `klaster_jabar_2025.geojson` belum disalin.
- Error saat instalasi di Cloud → pastikan `requirements.txt` ikut di-upload.

## D. Tangkapan layar untuk laporan
Gunakan `Windows + Shift + S` (Snipping Tool), mode jendela penuh, tampilan browser 100%. Disarankan:
1. **Ringkasan** → Gambar 4.12 (wajib).
2. **Hasil Klasterisasi** (metode Agglomerative Ward) → lampiran/gambar tambahan.
3. **Evaluasi & Perbandingan** (bagian TOPSIS interaktif).
4. **Learning Lab** → Playground.
Sertakan URL aplikasi pada keterangan gambar.
