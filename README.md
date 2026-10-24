# Pembuatan Ground Truth Estimasi Dimensi Lubang

Folder ini berisi seluruh skrip dan data untuk membuat ground truth (GT) dimensi lubang pada dataset primer (BNPD). Skrip diberi nomor sesuai urutan alur.

## Susunan folder

```
Pembuatan_GT/
├── 0_siapkan_folder.py          dijalankan sekali saja (menyalin foto dari folder eksperimen)
├── 1_build_pothole_gt.py        langkah 1: ukur 18 lubang dari foto tampak atas
├── 2_visualize_gt.py            langkah 2: gambar ukuran lubang di foto tampak atas
├── 3_build_validation_gt.py     langkah 3: petakan 100 anotasi POV kendaraan ke kode HOLE
├── 4_ipm_eksperimen.py          langkah 4: klik 4 titik acuan IPM per foto
├── 5_visualize_titik_acuan.py   langkah 5: gambar titik acuan + tabel lebar jalan
├── README.md
├── top_down/
│   ├── images/                  18 foto lubang tampak atas dengan kertas setengah A4
│   └── labels/                  bbox lubang (format YOLO)
├── real width/
│   ├── 3100/                    foto POV di ruas jalan selebar 3,10 m (24 foto)
│   ├── 4160/                    ruas 4,16 m (28 foto)
│   └── 4870/                    ruas 4,87 m (17 foto)
├── pothole_gt.csv               OUTPUT langkah 1: ukuran asli 18 lubang (HOLE_001 ... HOLE_018)
├── visualisasi_gt/              OUTPUT langkah 2
├── ipm_measurements.csv         OUTPUT langkah 4 (hasil ukur di BEV saat kalibrasi)
├── visualisasi_titik_acuan_ipm/ OUTPUT langkah 5
├── lebar_jalan_per_citra.csv    OUTPUT langkah 5
└── dataset_final/
    ├── validation_gt.csv        OUTPUT langkah 3: 100 anotasi -> kode HOLE + ukuran asli
    └── pov_kendaraan/
        ├── images/              70 foto POV kendaraan (69 berisi lubang, 1 citra negatif)
        ├── labels/              100 bbox (format YOLO)
        └── src_pts.csv          OUTPUT langkah 4: 4 titik acuan, lebar jalan, ukuran BEV
```

## Persiapan (sekali saja)

```
pip install opencv-python numpy pillow pandas
cd D:\Pembuatan_GT
python 0_siapkan_folder.py
```

Skrip 0 menyalin foto dan label dari `D:\data lokal\eksperimen` tanpa mengubah folder itu. Folder `real width/` dibangun ulang dari kolom `real_width_mm` di `src_pts.csv`, karena di folder eksperimen isi subfolder `4160` dan `4870` tertukar. Di akhir, skrip 0 menjalankan langkah 2 dan 5 untuk membuat visualisasi.

## Alur pembuatan GT

**Langkah 1. Ukuran asli lubang (foto tampak atas).** Di lapangan setiap lubang difoto dari atas bersama kertas setengah A4 (14,85 × 21,0 cm). Di jendela skrip, tarik garis di sepanjang sisi pendek kertas; kotak dengan rasio kertas akan terbentuk otomatis, lalu sesuaikan dengan scroll dan tombol A/D. Skrip menghitung `px_per_cm = lebar_kotak_px / 14,85`, lalu lebar dan panjang bbox lubang dibagi dengan nilai itu.

**Langkah 2. Visualisasi.** Hasil langkah 1 digambar di setiap foto tampak atas.

**Langkah 3. Pemetaan anotasi.** Setiap bbox di foto POV kendaraan dipasangkan dengan kode HOLE lewat thumbnail di panel kanan. Ukuran asli disalin dari `pothole_gt.csv`, sehingga satu lubang yang tampak di beberapa foto selalu punya ukuran GT yang sama.

**Langkah 4. Titik acuan IPM.** Untuk setiap foto, klik empat titik di jalan (TL, TR, BR, BL) yang membentuk persegi panjang selebar jalan dan sepanjang 5 m. Lebar jalan dibaca dari folder `real width/`. Ukuran BEV: lebar 600 px, tinggi `round(5000 × 600 / lebar_jalan_mm)`, yaitu 968 px (3,10 m), 721 px (4,16 m), dan 616 px (4,87 m).

**Langkah 5. Visualisasi titik acuan** dan tabel lebar jalan per foto.

## Memperbarui CSV yang sudah ada

Semua skrip interaktif membaca CSV yang sudah ada dan menawarkan pilihan mode. Sebelum menulis file, skrip selalu bertanya `Simpan ...? (y/N)`. Jawab `n` (atau langsung Enter) supaya data tidak berubah, misalnya saat demo.

| Skrip | Mode |
|---|---|
| 1 | `L` lanjut foto baru, `U` ukur ulang foto tertentu (kode HOLE tetap; bisa sekaligus menyinkronkan `validation_gt.csv`) |
| 3 | `L` lanjut, `U` ubah pemetaan foto tertentu (REAL_ID tetap), `S` sinkronkan ukuran dari `pothole_gt.csv` |
| 4 | `N` foto baru, `P` atur ulang titik foto tertentu, `A` semua foto, `W` perbarui lebar jalan dan `bev_h` dari folder `real width/` |
| 2, 5 | tidak interaktif, hasil lama ditimpa |

## Contoh demo saat sidang

1. `python 1_build_pothole_gt.py` → pilih `U` → pilih satu foto (misalnya HOLE_008) → gambar kotak di atas kertas → `C` → bandingkan dengan nilai lama yang dicetak di terminal → jawab `n`.
2. `python 3_build_validation_gt.py` → pilih `U` → pilih satu foto → tunjukkan pemetaan bbox ke HOLE → `N` → jawab `n`.
3. `python 4_ipm_eksperimen.py` → pilih `P` → pilih satu foto → titik lama dimuat, geser satu titik dan lihat BEV berubah → `C` → jawab `n`.

