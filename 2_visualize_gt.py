"""
2_visualize_gt.py
=================
LANGKAH 2. Visualisasi ukuran lubang hasil langkah 1.

Setiap foto di top_down/images digambar ulang dengan:
  - bounding box dari top_down/labels
  - kode HOLE, lebar, dan panjang (cm) dari pothole_gt.csv
Hasilnya disimpan ke visualisasi_gt/ (file lama ditimpa).

Jalankan:  python 2_visualize_gt.py
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path

# ── Konfigurasi path ────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
IMAGES_DIR   = BASE_DIR / "top_down" / "images"
LABELS_DIR   = BASE_DIR / "top_down" / "labels"
CSV_PATH     = BASE_DIR / "pothole_gt.csv"
OUTPUT_DIR   = BASE_DIR / "visualisasi_gt"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Warna & gaya ────────────────────────────────────────────────────────────
BOX_COLOR    = (0, 220, 100)      # hijau cerah (BGR)
TEXT_COLOR   = (255, 255, 255)    # putih
BG_COLOR     = (0, 0, 0)         # hitam (background label teks)
ID_COLOR     = (255, 200, 0)      # kuning (pothole id)
FONT         = cv2.FONT_HERSHEY_DUPLEX

# Resolusi referensi (diagonal piksel) — semua ukuran di-scale relatif ke ini
REF_DIAG_PX  = 640.0


def get_scale(img_w: int, img_h: int) -> dict:
    """
    Hitung parameter visual yang proporsional terhadap ukuran gambar.
    Semua nilai di-scale dari referensi diagonal REF_DIAG_PX (640 px).
    """
    import math
    diag   = math.sqrt(img_w ** 2 + img_h ** 2)
    ratio  = diag / REF_DIAG_PX          # >1 berarti gambar lebih besar
    return {
        "font_scale":   max(0.35, round(0.55 * ratio, 2)),
        "font_thick":   max(1, int(round(1 * ratio))),
        "box_thick":    max(2, int(round(2.5 * ratio))),
        "corner_len":   max(12, int(round(18 * ratio))),
        "pad":          max(4, int(round(6 * ratio))),
        "wm_scale":     max(0.25, round(0.38 * ratio, 2)),
    }

# ── Baca CSV ─────────────────────────────────────────────────────────────────
def load_csv(csv_path: Path) -> dict:
    """
    Kembalikan dict: image_filename -> list of hole_info
    Satu gambar bisa punya lebih dari satu pothole (gt_label_idx berbeda).
    """
    data: dict[str, list] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["gt_image"].strip()
            entry = {
                "pothole_id":  row["pothole_id"].strip(),
                "label_idx":   int(row["gt_label_idx"]),
                "yolo_cx":     float(row["yolo_cx"]),
                "yolo_cy":     float(row["yolo_cy"]),
                "yolo_w":      float(row["yolo_w"]),
                "yolo_h":      float(row["yolo_h"]),
                "gt_w_cm":     float(row["gt_w_cm"]),
                "gt_h_cm":     float(row["gt_h_cm"]),
                "gt_area_cm2": float(row["gt_area_cm2"]),
            }
            data.setdefault(img_name, []).append(entry)
    return data


# ── Konversi YOLO → piksel ───────────────────────────────────────────────────
def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


# ── Gambar teks dengan background ────────────────────────────────────────────
def put_text_with_bg(img, lines: list[str], origin_xy: tuple,
                     font=FONT, scale=0.55, thick=1,
                     text_color=TEXT_COLOR, bg_color=BG_COLOR, pad=6):
    """
    Tulis beberapa baris teks mulai dari origin_xy (x, y pojok kiri atas).
    Otomatis turun per baris. Dengan kotak background hitam semi-opak.
    Parameter scale, thick, pad sebaiknya diisi dari get_scale() agar
    konsisten terhadap resolusi gambar.
    """
    x0, y0 = origin_xy
    line_height = 0
    for line in lines:
        (tw, th), baseline = cv2.getTextSize(line, font, scale, thick)
        line_height = max(line_height, th + baseline)

    # total tinggi kotak
    total_h = line_height * len(lines) + pad * (len(lines) + 1)
    max_w   = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines) + pad * 2

    # clamp agar tidak keluar gambar
    img_h, img_w = img.shape[:2]
    x0 = min(x0, img_w - max_w - 2)
    y0 = min(y0, img_h - total_h - 2)
    x0, y0 = max(x0, 0), max(y0, 0)

    # gambar overlay semi-transparan
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + max_w, y0 + total_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # tulis teks
    y_cur = y0 + pad + line_height - 2
    for line in lines:
        cv2.putText(img, line, (x0 + pad, y_cur), font, scale, text_color, thick, cv2.LINE_AA)
        y_cur += line_height + pad


# ── Proses satu gambar ────────────────────────────────────────────────────────
def process_image(img_path: Path, label_path: Path, hole_infos: list, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] Tidak bisa baca gambar: {img_path.name}")
        return

    img_h, img_w = img.shape[:2]

    # Hitung semua parameter visual proporsional terhadap resolusi gambar
    sc = get_scale(img_w, img_h)
    font_scale = sc["font_scale"]
    font_thick = sc["font_thick"]
    box_thick  = sc["box_thick"]
    corner_len = sc["corner_len"]
    pad        = sc["pad"]
    wm_scale   = sc["wm_scale"]

    # Baca semua baris label YOLO (bisa >1 objek per gambar)
    yolo_rows = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                    yolo_rows.append((cls, cx, cy, w, h))

    # Gambar setiap bounding box
    for idx, (cls, cx, cy, w, h) in enumerate(yolo_rows):
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)

        # Cari data CSV yang cocok berdasarkan label_idx
        info = next((e for e in hole_infos if e["label_idx"] == idx), None)

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, box_thick)

        # Sudut kecil di pojok kotak (hiasan) — ukuran proporsional
        for cx2, cy2, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(img, (cx2, cy2), (cx2 + dx*corner_len, cy2), (255,255,255), max(1, font_thick))
            cv2.line(img, (cx2, cy2), (cx2, cy2 + dy*corner_len), (255,255,255), max(1, font_thick))

        if info:
            pid  = info["pothole_id"]
            w_cm = info["gt_w_cm"]
            h_cm = info["gt_h_cm"]
            fmt  = lambda v: f"{v:.2f}".replace(".", ",")
            lines = [
                f"{pid}",
                f"Lebar   : {fmt(w_cm)} cm",
                f"Panjang : {fmt(h_cm)} cm",
            ]
        else:
            lines = [f"class={cls} | idx={idx}", "(no CSV data)"]

        # Tempatkan teks di atas bounding box (atau di dalam jika terlalu ke atas)
        txt_y = max(y1 - pad, pad)
        put_text_with_bg(img, lines, (x1, txt_y),
                         scale=font_scale, thick=font_thick, pad=pad)

    # Watermark nama file di pojok kiri bawah
    wm = img_path.stem[:60]
    cv2.putText(img, wm, (pad, img_h - pad),
                FONT, wm_scale, (180, 180, 180), max(1, font_thick - 1), cv2.LINE_AA)

    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  [OK] {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  LANGKAH 2  -  VISUALISASI GT TAMPAK ATAS")
    print(f"  Images  : {IMAGES_DIR}")
    print(f"  Labels  : {LABELS_DIR}")
    print(f"  CSV     : {CSV_PATH}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 60)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {CSV_PATH}")

    csv_data = load_csv(CSV_PATH)
    print(f"\nLoaded {sum(len(v) for v in csv_data.values())} entri dari CSV\n")

    img_files = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if not img_files:
        print("Tidak ada gambar ditemukan di folder images/")
        return

    for img_path in img_files:
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        hole_infos = csv_data.get(img_path.name, [])
        out_path   = OUTPUT_DIR / img_path.name

        n_holes = len(hole_infos)
        print(f"Proses: {img_path.name}  [{n_holes} hole(s) di CSV]")
        process_image(img_path, label_path, hole_infos, out_path)

    print(f"\n[DONE] Selesai! {len(img_files)} gambar disimpan ke: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
