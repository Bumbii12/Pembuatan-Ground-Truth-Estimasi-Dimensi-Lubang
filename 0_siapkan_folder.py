"""
0_siapkan_folder.py
===================
Dijalankan SEKALI untuk mengisi folder Pembuatan_GT dengan foto dan label
dari folder eksperimen lama. Folder lama tidak diubah sama sekali (hanya
disalin), dan file yang sudah ada di Pembuatan_GT tidak ditimpa.

Yang disalin:
  eksperimen/gt/images, gt/labels  -> top_down/images, top_down/labels
  eksperimen/images, labels        -> dataset_final/pov_kendaraan/images, labels
  foto POV ke 'real width/<lebar>/' sesuai kolom real_width_mm di
  dataset_final/pov_kendaraan/src_pts.csv (lebar yang dipakai naskah).
  Catatan: di folder eksperimen, isi subfolder 4160 dan 4870 tertukar,
  jadi folder 'real width/' dibangun ulang dari src_pts.csv, bukan disalin.

Setelah selesai, skrip menjalankan langkah 2 dan 5 untuk membuat ulang
visualisasi_gt/, visualisasi_titik_acuan_ipm/, dan lebar_jalan_per_citra.csv.

Jalankan:  python 0_siapkan_folder.py
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SUMBER = Path(r"D:\data lokal\eksperimen")
TUJUAN = Path(__file__).parent

POV_DIR   = TUJUAN / "dataset_final" / "pov_kendaraan"
SRC_PTS   = POV_DIR / "src_pts.csv"
RW_DIR    = TUJUAN / "real width"


def salin_folder(src, dst, pola=("*.jpg", "*.jpeg", "*.png", "*.txt")):
    dst.mkdir(parents=True, exist_ok=True)
    n_baru = n_ada = 0
    for p in pola:
        for f in sorted(src.glob(p)):
            target = dst / f.name
            if target.exists():
                n_ada += 1
                continue
            shutil.copy2(f, target)
            n_baru += 1
    print(f"   {src.relative_to(SUMBER)}  ->  {dst.relative_to(TUJUAN)}   "
          f"({n_baru} disalin, {n_ada} sudah ada)")


def main():
    if not SUMBER.exists():
        print(f"❌  Folder sumber tidak ditemukan: {SUMBER}")
        sys.exit(1)
    if not SRC_PTS.exists():
        print(f"❌  {SRC_PTS} tidak ditemukan. Pastikan src_pts.csv sudah ada.")
        sys.exit(1)

    print("=" * 60)
    print("  MENYIAPKAN FOLDER Pembuatan_GT")
    print("=" * 60)
    print(f"Sumber : {SUMBER}")
    print(f"Tujuan : {TUJUAN}\n")

    print("1) Foto dan label")
    salin_folder(SUMBER / "gt" / "images", TUJUAN / "top_down" / "images")
    salin_folder(SUMBER / "gt" / "labels", TUJUAN / "top_down" / "labels")
    salin_folder(SUMBER / "images", POV_DIR / "images")
    salin_folder(SUMBER / "labels", POV_DIR / "labels")

    print("\n2) Folder 'real width/' dari src_pts.csv")
    with open(SRC_PTS, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    hitung = {}
    for r in rows:
        lebar = str(int(float(r["real_width_mm"])))
        src = POV_DIR / "images" / r["image"]
        dst = RW_DIR / lebar / r["image"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"   ⚠  Foto tidak ditemukan: {r['image']}")
            continue
        if not dst.exists():
            shutil.copy2(src, dst)
        hitung[lebar] = hitung.get(lebar, 0) + 1
    for lebar in sorted(hitung):
        print(f"   real width/{lebar}: {hitung[lebar]} foto")

    # Pemeriksaan: setiap foto hanya ada di satu subfolder dan cocok dengan src_pts
    salah = []
    lebar_csv = {r["image"]: str(int(float(r["real_width_mm"]))) for r in rows}
    for sub in RW_DIR.iterdir():
        if sub.is_dir():
            for f in sub.iterdir():
                if lebar_csv.get(f.name) != sub.name:
                    salah.append(f"{sub.name}/{f.name}")
    if salah:
        print("   ⚠  Foto di subfolder yang tidak sesuai src_pts.csv:")
        for s in salah:
            print(f"      {s}")
    else:
        print("   ✅  Semua foto di 'real width/' sesuai dengan src_pts.csv")

    print("\n3) Membuat visualisasi (langkah 2 dan 5)")
    for skrip in ("2_visualize_gt.py", "5_visualize_titik_acuan.py"):
        print(f"\n--- {skrip} ---", flush=True)
        subprocess.run([sys.executable, str(TUJUAN / skrip)], cwd=TUJUAN, check=False)

    print("\nSelesai. Folder siap dipakai. Lihat README.md untuk langkah demo.")


if __name__ == "__main__":
    main()
