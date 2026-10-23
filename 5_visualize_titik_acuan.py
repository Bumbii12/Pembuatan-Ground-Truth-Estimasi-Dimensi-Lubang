"""
5_visualize_titik_acuan.py
==========================
LANGKAH 5. Visualisasi empat titik acuan IPM (TL, TR, BR, BL) pada setiap
foto POV kendaraan, sekaligus membuat tabel ringkasan lebar jalan per foto.

Masukan:
  - dataset_final/pov_kendaraan/images/  : foto POV kendaraan
  - dataset_final/pov_kendaraan/src_pts.csv
  - dataset_final/validation_gt.csv      : untuk kolom kode_lubang
Keluaran (file lama ditimpa):
  - visualisasi_titik_acuan_ipm/<nama_foto>_titik_acuan.jpg
  - lebar_jalan_per_citra.csv

Titik yang berada di luar bingkai foto (misalnya BL dengan x negatif) tetap
ditampilkan dengan memperlebar kanvas berlatar abu-abu.

Jalankan:  python 5_visualize_titik_acuan.py
Butuh: pip install pandas pillow
"""

from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ── Konfigurasi path ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
POV_DIR    = BASE_DIR / "dataset_final" / "pov_kendaraan"
IMAGES_DIR = POV_DIR / "images"
CSV_PATH   = POV_DIR / "src_pts.csv"
VAL_CSV    = BASE_DIR / "dataset_final" / "validation_gt.csv"
OUTPUT_DIR = BASE_DIR / "visualisasi_titik_acuan_ipm"
TABLE_CSV  = BASE_DIR / "lebar_jalan_per_citra.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SIDE   = 1800                    # sisi terpanjang gambar keluaran (px)

# ── Warna ───────────────────────────────────────────────────────────────────
BG_CANVAS  = (235, 235, 235)         # latar kanvas di luar foto
AREA_FILL  = (255, 200, 0, 55)       # isi area acuan (RGBA, semi-transparan)
AREA_LINE  = (255, 200, 0)           # garis area acuan
POINT_FILL = (220, 30, 30)           # titik acuan
INFO_BG    = (20, 20, 20)            # latar kotak informasi


def load_font(size, bold=False):
    """Cari font yang tersedia (Windows atau Linux), fallback ke default."""
    candidates = (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
                  else ["arial.ttf", "DejaVuSans.ttf"])
    for name in candidates:
        for path in (name,
                     f"C:/Windows/Fonts/{name}",
                     f"/usr/share/fonts/truetype/dejavu/{name}"):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def fmt(v):
    """Angka dua desimal dengan koma (format Indonesia)."""
    return f"{v:.2f}".replace(".", ",")


def process(row):
    img_path = IMAGES_DIR / row.image
    if not img_path.exists():
        print(f"  [SKIP] Citra tidak ditemukan: {row.image}")
        return

    im = Image.open(img_path).convert("RGB")
    W, H = im.size

    pts = {
        "TL": (int(row.tl_x), int(row.tl_y)),
        "TR": (int(row.tr_x), int(row.tr_y)),
        "BR": (int(row.br_x), int(row.br_y)),
        "BL": (int(row.bl_x), int(row.bl_y)),
    }

    # Kanvas diperlebar agar titik di luar bingkai foto tetap terlihat
    xs = [p[0] for p in pts.values()] + [0, W]
    ys = [p[1] for p in pts.values()] + [0, H]
    m = int(0.04 * W)
    x0, y0 = min(xs) - m, min(ys) - m
    x1, y1 = max(xs) + m, max(ys) + m

    canvas = Image.new("RGB", (x1 - x0, y1 - y0), BG_CANVAS)
    canvas.paste(im, (-x0, -y0))
    thick = max(4, W // 300)

    # Bingkai foto asli
    d = ImageDraw.Draw(canvas)
    d.rectangle([-x0, -y0, -x0 + W, -y0 + H], outline=(120, 120, 120), width=thick // 2)

    # Area acuan (isi semi-transparan + garis tepi)
    poly = [(x - x0, y - y0) for x, y in pts.values()]
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).polygon(poly, fill=AREA_FILL)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(canvas)
    d.line(poly + [poly[0]], fill=AREA_LINE, width=thick)

    # Titik acuan + label koordinat
    rad = int(W / 80)
    f_lab = load_font(int(W / 32), bold=True)
    for name, (x, y) in pts.items():
        X, Y = x - x0, y - y0
        d.ellipse([X - rad, Y - rad, X + rad, Y + rad],
                  fill=POINT_FILL, outline="white", width=thick // 2)
        label = f"{name} ({x}, {y})"
        tw = d.textlength(label, font=f_lab)
        oy = -rad * 4 if name in ("BL", "BR") else rad * 1.5
        tx = min(max(X - tw / 2, 5), canvas.width - tw - 5)
        d.text((tx, Y + oy), label, fill="black", font=f_lab,
               stroke_width=thick, stroke_fill="white")

    # Kotak informasi di pojok kiri bawah foto
    f_info = load_font(int(W / 30))
    info = [
        row.image.split("_jpg")[0],
        f"Lebar jalan  : {fmt(row.real_width_mm / 1000)} m",
        f"Panjang area : {fmt(row.real_depth_mm / 1000)} m",
        f"BEV : {row.bev_w} x {row.bev_h} px",
    ]
    lh = int(W / 30 * 1.35)
    bw = max(d.textlength(l, font=f_info) for l in info) + 40
    bx, by = -x0 + 20, -y0 + H - lh * len(info) - 40
    d.rectangle([bx, by, bx + bw, by + lh * len(info) + 20], fill=INFO_BG)
    for i, line in enumerate(info):
        d.text((bx + 20, by + 10 + i * lh), line, fill="white", font=f_info)

    canvas.thumbnail((MAX_SIDE, MAX_SIDE))
    out = OUTPUT_DIR / f"{row.image.split('_jpg')[0]}_titik_acuan.jpg"
    canvas.save(out, quality=90)
    print(f"  [OK] {out.name}")


def write_table(df):
    """Tabel ringkasan: tanggal, jam, lebar jalan, ukuran BEV, dan kode lubang per foto."""
    holes = {}
    if VAL_CSV.exists():
        val = pd.read_csv(VAL_CSV).sort_values(["real_image", "pothole_idx_in_image"])
        for img, grp in val.groupby("real_image"):
            holes[img] = ", ".join(sorted(set(grp["pothole_id"])))
    rows = []
    for r in df.itertuples():
        stamp = r.image.replace("IMG_", "").split("_jpg")[0]      # 20260530_230110
        d, t = stamp.split("_")[:2]
        rows.append({
            "image":          r.image,
            "tanggal":        f"{d[:4]}-{d[4:6]}-{d[6:8]}",
            "waktu_wib":      f"{t[:2]}:{t[2:4]}:{t[4:6]}",
            "lebar_jalan_m":  r.real_width_mm / 1000,
            "panjang_area_m": r.real_depth_mm / 1000,
            "bev_w_px":       r.bev_w,
            "bev_h_px":       r.bev_h,
            "kode_lubang":    holes.get(r.image, ""),
        })
    out = pd.DataFrame(rows).sort_values(["tanggal", "waktu_wib"])
    out.to_csv(TABLE_CSV, index=False)
    print(f"[OK] Tabel lebar jalan: {TABLE_CSV.name} ({len(out)} foto)")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"{len(df)} foto di {CSV_PATH.name}\n")
    for row in df.itertuples():
        process(row)
    print(f"\n[DONE] Visualisasi disimpan ke: {OUTPUT_DIR}")
    write_table(df)


if __name__ == "__main__":
    main()
