"""
4_ipm_eksperimen.py
===================
LANGKAH 4. Kalibrasi IPM (Inverse Perspective Mapping) untuk setiap foto
POV kendaraan.

Untuk setiap foto, pengguna mengklik empat titik acuan di jalan yang
membentuk persegi panjang nyata dengan:
  - lebar  = lebar jalan, dibaca dari nama subfolder di 'real width/'
             (3100, 4160, 4870 mm)
  - panjang = 5000 mm (sama untuk semua foto)
Dari empat titik itu dibuat matriks homografi ke tampak atas (BEV) selebar
600 px, dengan tinggi BEV = round(5000 x 600 / lebar_jalan).

Alur per foto:
  1. Klik 4 titik: TL (kiri-jauh) -> TR (kanan-jauh) -> BR (kanan-dekat) -> BL (kiri-dekat)
     Titik BOLEH diklik di LUAR batas foto (area abu-abu di sekeliling foto)
  2. Pratinjau BEV tampil langsung setelah 4 titik
  3. Bbox lubang diproyeksikan ke BEV dan diukur
     (lebar = sisi bawah, panjang = rata-rata sisi kiri dan kanan)
  4. C/Enter simpan, S skip, Z undo titik terakhir

Output:
  - dataset_final/pov_kendaraan/src_pts.csv -> 4 titik acuan, lebar jalan, ukuran BEV
  - ipm_measurements.csv                    -> hasil ukur lubang di BEV (cm)

Mode saat src_pts.csv sudah ada:
  [N] Baru   : hanya foto yang belum dikalibrasi (default)
  [P] Pilih  : pilih foto tertentu untuk diatur ulang titiknya
  [A] Semua  : atur ulang titik semua foto
  [W] Lebar  : perbarui real_width_mm dan bev_h dari folder 'real width/'
               tanpa klik ulang (bev_h ikut dihitung ulang)
  Sebelum menulis file, skrip selalu meminta konfirmasi (y/N).

Kontrol:
  Klik kiri   -> Tambah titik baru (maks 4), bisa di luar foto
  Klik+Geser  -> Geser titik yang sudah ada
  Z           -> Hapus titik terakhir (undo)
  R           -> Reset semua titik
  P           -> Cetak hasil ukur saat ini ke terminal
  F / 1       -> Zoom pas layar / zoom 1:1
  C / Enter   -> Konfirmasi & simpan (perlu 4 titik)
  S           -> Skip foto ini
  Q / Esc     -> Keluar
"""

# -*- coding: utf-8 -*-
import cv2
import numpy as np
import csv
import sys
from pathlib import Path

# Paksa stdout ke UTF-8 agar emoji/unicode tidak error di terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── KONFIGURASI FISIK ────────────────────────────────────────────────────────
# real_width_mm dibaca per-gambar dari subfolder di folder 'real width/'
# (nama subfolder = nilai lebar jalan dalam mm, misal: 3100, 4160, 4870)
DEFAULT_WIDTH_MM = 4016   # fallback jika gambar tidak ditemukan di folder real width/
REAL_DEPTH_MM    = 5000   # kedalaman jalan nyata (mm) — berlaku untuk semua gambar

# BEV canvas: lebar tetap 600px, tinggi dihitung proporsional terhadap depth
# (BEV_W/BEV_H/SCALE akan dihitung ulang per-gambar berdasarkan real_width_mm-nya)
BEV_W = 600

# Globals yang di-update per-gambar di dalam main()
# Nilai awal = placeholder berdasarkan DEFAULT; akan di-overwrite sebelum dipakai
REAL_WIDTH_MM = DEFAULT_WIDTH_MM
BEV_H         = round(REAL_DEPTH_MM * BEV_W / DEFAULT_WIDTH_MM)
SCALE         = DEFAULT_WIDTH_MM / BEV_W

# ─── PATH ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
POV_DIR        = BASE_DIR / "dataset_final" / "pov_kendaraan"
SRC_PTS_CSV    = POV_DIR / "src_pts.csv"
MEAS_CSV       = BASE_DIR / "ipm_measurements.csv"
REALWIDTH_DIR  = BASE_DIR / "real width"            # folder klasifikasi lebar jalan

# Folder gambar dan label POV kendaraan
IMAGES_DIR = POV_DIR / "images"
LABELS_DIR = POV_DIR / "labels"
IMG_EXTS   = {".jpg", ".jpeg", ".png"}


def build_realwidth_map():
    """
    Baca folder 'real width/' dan bangun dict {nama_file: real_width_mm}.
    Setiap subfolder bernama angka (misal: 3100, 4160, 4870) berisi file gambar.
    Gambar yang tidak ditemukan → DEFAULT_WIDTH_MM.
    """
    mapping = {}
    if not REALWIDTH_DIR.exists():
        print(f"⚠   Folder '{REALWIDTH_DIR.name}' tidak ditemukan — semua gambar pakai DEFAULT_WIDTH_MM={DEFAULT_WIDTH_MM}")
        return mapping
    for subfolder in REALWIDTH_DIR.iterdir():
        if not subfolder.is_dir():
            continue
        try:
            width_mm = int(subfolder.name)
        except ValueError:
            print(f"⚠   Subfolder '{subfolder.name}' bukan angka, dilewati.")
            continue
        for f in subfolder.iterdir():
            if f.suffix.lower() in IMG_EXTS:
                mapping[f.name] = width_mm
    return mapping

# ─── LAYOUT WINDOW ────────────────────────────────────────────────────────────
WINDOW_ORIG = "IPM — Gambar Asli (klik 4 titik)"
WINDOW_BEV  = "IPM — Hasil BEV"

# Padding (px) di sekeliling gambar — area ini bisa diklik untuk titik di luar gambar
# Naikkan nilai ini jika butuh lebih banyak ruang di luar gambar
PADDING = 3000

# Radius SCREEN px — klik dalam radius ini di sekitar titik = drag
# (40px = cukup besar, mudah diklik bahkan saat zoom out)
DRAG_RADIUS = 40

# Zoom batas dan step per scroll-tick
ZOOM_MIN  = 0.15
ZOOM_MAX  = 12.0
ZOOM_STEP = 1.25   # faktor zoom per satu scroll

# Ukuran maksimum display window (px) — canvas bisa lebih besar, tapi yang ditampilkan
# dibatasi agar tidak meluber dari layar; zoom/pan untuk menjelajah sisanya
DISP_MAX_W = 1600
DISP_MAX_H = 1000

# Warna area padding
PAD_BG_COLOR = (28, 28, 38)   # biru-gelap gelap

# Warna titik kalibrasi per-urutan
PT_COLORS = [
    (0, 255, 255),   # TL = kuning-cyan
    (0, 165, 255),   # TR = oranye
    (0, 0, 255),     # BR = merah
    (255, 0, 255),   # BL = magenta
]
PT_LABELS = ["TL (kiri-jauh)", "TR (kanan-jauh)", "BR (kanan-dekat)", "BL (kiri-dekat)"]

# ─── HELPER ───────────────────────────────────────────────────────────────────

def load_yolo_labels(label_path):
    """Baca label YOLO .txt → list [cx, cy, bw, bh] (normalized)."""
    boxes = []
    p = Path(label_path)
    if p.exists():
        for line in p.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                boxes.append([float(x) for x in parts[1:5]])
    return boxes


def build_homography(pts_4, bev_w, bev_h):
    """
    Bangun matriks homografi dari 4 titik sumber ke BEV rectangle.
    pts_4: list of 4 (x,y) dalam koordinat GAMBAR (bisa negatif = di luar gambar)
    """
    src = np.float32(pts_4)
    dst = np.float32([
        [0,     0    ],   # TL
        [bev_w, 0    ],   # TR
        [bev_w, bev_h],   # BR
        [0,     bev_h],   # BL
    ])
    return cv2.getPerspectiveTransform(src, dst)


def project_point(pt, M):
    """Proyeksikan satu titik (x, y) melalui homografi M ke BEV."""
    p = np.float32([[pt]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, M).reshape(2)


def project_bbox_to_bev(cx, cy, bw, bh, img_w, img_h, M):
    """
    Proyeksikan bbox YOLO (normalized) ke BEV.
    Return: 4 sudut di BEV space (np.array shape (4,2)),
            urutan: TL, TR, BR, BL (top = jauh, bottom = dekat kamera)
    """
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    corners = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]).reshape(-1,1,2)
    bev_corners = cv2.perspectiveTransform(corners, M).reshape(-1,2)
    return bev_corners  # [TL_bev, TR_bev, BR_bev, BL_bev]


def trapezoid_ratio(bev_corners):
    """
    Hitung rasio lebar atas vs bawah di BEV.
    Mendekati 1.0 = baik (persegi panjang).
    Jauh dari 1.0 = distorsi trapesium besar → hasil kurang akurat.
    bev_corners urutan: TL, TR, BR, BL
    """
    tl, tr, br, bl = bev_corners
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    if w_bot < 1e-6:
        return 999.0
    return w_top / w_bot


def measure_pothole_bottom_edge(bev_corners, scale_mm_px):
    """
    Pengukuran pothole di BEV.
    - LEBAR  : tepi BAWAH bbox di BEV (paling dekat kamera = paling akurat)
    - TINGGI : rata-rata tinggi sisi kiri dan sisi kanan trapesium
    - AREA   : W × H
    bev_corners urutan: TL, TR, BR, BL
    """
    tl, tr, br, bl = bev_corners
    w_bot    = np.linalg.norm(br - bl)
    h_left   = np.linalg.norm(bl - tl)
    h_right  = np.linalg.norm(br - tr)
    h_avg    = (h_left + h_right) / 2
    w_mm     = round(w_bot * scale_mm_px, 1)
    h_mm     = round(h_avg * scale_mm_px, 1)
    area_mm2 = round(w_mm * h_mm, 0)
    return w_mm, h_mm, area_mm2


def is_in_bev(bev_corners, bev_w, bev_h, margin=10):
    """Cek apakah semua sudut bbox masuk dalam canvas BEV."""
    return all(
        -margin <= c[0] <= bev_w + margin and
        -margin <= c[1] <= bev_h + margin
        for c in bev_corners
    )


# ─── CANVAS BUILDER ───────────────────────────────────────────────────────────

def make_padded_canvas(img, padding):
    """
    Buat canvas berpadding di sekeliling gambar.
    Gambar asli diletakkan di tengah. Area padding = abu-abu gelap.
    Return: canvas (numpy array BGR)
    """
    h, w = img.shape[:2]
    ch = h + 2 * padding
    cw = w + 2 * padding
    canvas = np.full((ch, cw, 3), PAD_BG_COLOR, dtype=np.uint8)

    # Gambar grid tipis di area padding sebagai panduan visual
    grid_step = 50
    grid_color = (45, 45, 60)
    for gx in range(0, cw, grid_step):
        cv2.line(canvas, (gx, 0), (gx, ch), grid_color, 1)
    for gy in range(0, ch, grid_step):
        cv2.line(canvas, (0, gy), (cw, gy), grid_color, 1)

    # Tempel gambar asli
    canvas[padding:padding+h, padding:padding+w] = img

    # Border tegas sekeliling gambar
    cv2.rectangle(canvas,
                  (padding - 1, padding - 1),
                  (padding + w, padding + h),
                  (80, 80, 100), 2)

    # Label sudut gambar agar mudah orientasi
    corner_labels = [
        ((padding + 4,      padding + 14),      "TL gambar"),
        ((padding + w - 70, padding + 14),      "TR gambar"),
        ((padding + w - 70, padding + h - 4),   "BR gambar"),
        ((padding + 4,      padding + h - 4),   "BL gambar"),
    ]
    for pos, txt in corner_labels:
        cv2.putText(canvas, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (70, 70, 90), 1, cv2.LINE_AA)

    return canvas


def img_to_canvas(pt_img, padding):
    """Konversi koordinat gambar → koordinat canvas."""
    return (int(pt_img[0]) + padding, int(pt_img[1]) + padding)


def canvas_to_img(cx, cy, padding):
    """Konversi koordinat canvas → koordinat gambar (bisa negatif)."""
    return (cx - padding, cy - padding)


def draw_orig(img, cal_pts, next_label, padding, dragging_idx=-1, hover_idx=-1):
    """
    Gambar titik kalibrasi dan instruksi di atas canvas berpadding.
    cal_pts      : list koordinat dalam sistem GAMBAR (bisa negatif)
    dragging_idx : index titik yang sedang di-drag (-1 = tidak ada)
    hover_idx    : index titik yang kursor berada di atasnya (-1 = tidak ada)
    """
    canvas = make_padded_canvas(img, padding)
    ch, cw = canvas.shape[:2]

    # Garis penghubung antar titik (gambar dulu supaya di bawah titik)
    if len(cal_pts) >= 2:
        canvas_pts = [img_to_canvas(p, padding) for p in cal_pts]
        for i in range(len(canvas_pts) - 1):
            cv2.line(canvas, canvas_pts[i], canvas_pts[i+1], (180, 180, 180), 1, cv2.LINE_AA)
        if len(cal_pts) == 4:
            cv2.line(canvas, canvas_pts[3], canvas_pts[0], (180, 180, 180), 1, cv2.LINE_AA)

    # Gambar setiap titik
    for i, pt_img in enumerate(cal_pts):
        pt_c        = img_to_canvas(pt_img, padding)
        color       = PT_COLORS[i]
        is_dragging = (i == dragging_idx)
        is_hover    = (i == hover_idx and not is_dragging)

        if is_dragging:
            # Sedang di-drag: lingkaran putih besar + aura + crosshair
            cv2.circle(canvas, pt_c, DRAG_RADIUS + 10, color, 1, cv2.LINE_AA)  # aura tipis
            cv2.circle(canvas, pt_c, 14, (255, 255, 255), -1)
            cv2.circle(canvas, pt_c, 14, color, 3, cv2.LINE_AA)
            arm = 26
            cv2.line(canvas, (pt_c[0]-arm, pt_c[1]), (pt_c[0]+arm, pt_c[1]), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (pt_c[0], pt_c[1]-arm), (pt_c[0], pt_c[1]+arm), color, 2, cv2.LINE_AA)
        elif is_hover:
            # Hover: glowing ring + titik lebih besar (siap di-grab)
            cv2.circle(canvas, pt_c, DRAG_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)  # outer ring putih
            cv2.circle(canvas, pt_c, DRAG_RADIUS - 4, color, 1, cv2.LINE_AA)        # inner ring
            cv2.circle(canvas, pt_c, 11, color, -1)                                  # titik isi
            cv2.circle(canvas, pt_c, 13, (255, 255, 255), 2)                          # border putih
            # Label "GRAB" kecil
            cv2.putText(canvas, "GRAB", (pt_c[0] + 16, pt_c[1] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "GRAB", (pt_c[0] + 16, pt_c[1] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        else:
            # Normal: titik kecil + ring drag-zone tipis (hint area)
            cv2.circle(canvas, pt_c, DRAG_RADIUS, color, 1, cv2.LINE_AA)  # zona drag
            cv2.circle(canvas, pt_c, 8, color, -1)
            cv2.circle(canvas, pt_c, 10, (255, 255, 255), 2)

        # Label titik
        label_pos = (pt_c[0] + 14, pt_c[1] - 10)
        (lw, lh), _ = cv2.getTextSize(PT_LABELS[i], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.rectangle(canvas,
                      (label_pos[0] - 2, label_pos[1] - lh - 2),
                      (label_pos[0] + lw + 2, label_pos[1] + 2),
                      (0, 0, 0), -1)
        cv2.putText(canvas, PT_LABELS[i], label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

        # Koordinat kecil di bawah titik
        coord_str = f"({pt_img[0]}, {pt_img[1]})"
        cv2.putText(canvas, coord_str, (pt_c[0] + 14, pt_c[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    # ── Instruksi bar atas ──────────────────────────────────────────────────
    if dragging_idx >= 0:
        dragged_name = PT_LABELS[dragging_idx]
        instr       = f"  DRAG: geser [{dragged_name}] ke posisi baru — lepas klik untuk drop"
        color_instr = PT_COLORS[dragging_idx]
    elif len(cal_pts) < 4:
        instr       = f"  Klik titik {len(cal_pts)+1}/4: {next_label}  |  Drag titik lama untuk geser"
        color_instr = PT_COLORS[len(cal_pts)]
    else:
        instr       = "  BEV live di kanan |↻ Drag titik = update real-time | C/Enter: simpan | R: reset | Z: undo | P: print ukuran"
        color_instr = (50, 255, 50)

    cv2.rectangle(canvas, (0, 0), (cw, 34), (0, 0, 0), -1)
    cv2.putText(canvas, instr, (4, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_instr, 2, cv2.LINE_AA)

    # ── Info bar bawah ──────────────────────────────────────────────────────
    info = (f"  Fisik: {REAL_WIDTH_MM}mm x {REAL_DEPTH_MM}mm  |  "
            f"BEV: {BEV_W}x{BEV_H}px  |  Scale: {SCALE:.2f}mm/px  |  "
            f"[Klik area abu-abu = titik di luar gambar  |  Klik+geser = pindah titik]")
    cv2.rectangle(canvas, (0, ch - 26), (cw, ch), (0, 0, 0), -1)
    cv2.putText(canvas, info, (4, ch - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 160), 1, cv2.LINE_AA)

    return canvas


def draw_bev(M, img, boxes, img_w, img_h):
    """Warp gambar ke BEV dan gambar pothole boxes + ukuran."""
    bev = cv2.warpPerspective(img, M, (BEV_W, BEV_H))

    # Grid referensi (setiap 100mm)
    grid_step_mm = 100
    grid_step_px = int(grid_step_mm / SCALE)
    for gx in range(0, BEV_W, grid_step_px):
        cv2.line(bev, (gx, 0), (gx, BEV_H), (40, 40, 40), 1)
    for gy in range(0, BEV_H, grid_step_px):
        cv2.line(bev, (0, gy), (BEV_W, gy), (40, 40, 40), 1)

    # Label grid
    for i, gx in enumerate(range(0, BEV_W, grid_step_px)):
        cv2.putText(bev, f"{i*grid_step_mm}mm", (gx+2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)

    # Proyeksikan dan ukur setiap pothole
    measurements = []
    for idx, box in enumerate(boxes):
        cx, cy, bw_n, bh_n = box[:4]
        bev_corners = project_bbox_to_bev(cx, cy, bw_n, bh_n, img_w, img_h, M)

        if not is_in_bev(bev_corners, BEV_W, BEV_H):
            measurements.append(None)
            continue

        t_ratio = trapezoid_ratio(bev_corners)

        if t_ratio < 1.2:
            box_color  = (0, 220, 0)
            fill_color = (0, 60, 0)
            reliability = "OK"
        elif t_ratio < 1.8:
            box_color  = (0, 180, 255)
            fill_color = (0, 50, 80)
            reliability = "~"
        else:
            box_color  = (0, 0, 220)
            fill_color = (60, 0, 0)
            reliability = "!"

        w_mm, h_mm, area_mm2 = measure_pothole_bottom_edge(bev_corners, SCALE)

        measurements.append({
            "pothole_idx":    idx + 1,
            "w_mm":           w_mm,
            "h_mm":           h_mm,
            "area_mm2":       area_mm2,
            "w_cm":           round(w_mm/10, 2),
            "h_cm":           round(h_mm/10, 2),
            "area_cm2":       round(area_mm2/100, 2),
            "trapezoid_ratio": round(t_ratio, 3),
            "reliability":    reliability,
        })

        pts_int = bev_corners.astype(np.int32)
        overlay = bev.copy()
        cv2.fillPoly(overlay, [pts_int], fill_color)
        cv2.addWeighted(overlay, 0.5, bev, 0.5, 0, bev)
        cv2.polylines(bev, [pts_int], isClosed=True, color=box_color, thickness=2)

        tl_b, tr_b, br_b, bl_b = bev_corners.astype(np.int32)
        cv2.line(bev, tuple(br_b), tuple(bl_b), (255, 255, 0), 2)

        cx_bev = int(bev_corners[:, 0].mean())
        cy_bev = int(bev_corners[:, 1].mean())
        lbl1 = f"#{idx+1} {reliability}: {w_mm:.0f}x{h_mm:.0f}mm"
        lbl2 = f"  {area_mm2/100:.1f}cm2  (trap={t_ratio:.2f})"
        cv2.putText(bev, lbl1, (cx_bev - 55, cy_bev - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(bev, lbl2, (cx_bev - 55, cy_bev + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1)

    # Legend
    cv2.putText(bev, "Hijau=OK  Orng=sedang  Merah=trapezoid!",
                (4, BEV_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)
    cv2.putText(bev, "Kuning = tepi bawah bbox (paling akurat)",
                (4, BEV_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 220, 220), 1)
    cv2.putText(bev, f"<-- {REAL_WIDTH_MM}mm -->",
                (BEV_W//2 - 40, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 180, 255), 1)

    return bev, measurements


# build_combined_view dihapus — BEV kini tampil di window terpisah (WINDOW_BEV)


# ─── ZOOM/PAN + MOUSE STATE ──────────────────────────────────────────────────

cal_pts      = []    # koordinat titik dalam sistem GAMBAR (bisa negatif)
dragging_idx = -1    # index titik yang sedang di-drag; -1 = tidak ada

view_zoom  = 1.0     # zoom saat ini (1.0 = 1:1)
view_pan_x = 0.0     # top-left viewport dalam koordinat logical canvas (x)
view_pan_y = 0.0     # top-left viewport dalam koordinat logical canvas (y)

panning       = False   # sedang pan (klik-kanan + geser)?
pan_start_s   = (0, 0)  # posisi screen saat pan dimulai
pan_start_p   = (0.0, 0.0)  # view_pan saat pan dimulai
hover_idx     = -1      # index titik yang kursor berada di atasnya; -1 = tidak ada
_bev_last_pts = None   # cache tuple cal_pts terakhir, untuk throttle BEV recompute

# ─── ZOOM/PAN HELPERS ─────────────────────────────────────────────────────────

def screen_to_logical(sx, sy):
    """Konversi koordinat screen (px di window) → koordinat logical canvas."""
    return view_pan_x + sx / view_zoom, view_pan_y + sy / view_zoom


def render_view(logical_canvas, disp_w, disp_h):
    """
    Render logical canvas ke display window menggunakan view_zoom / view_pan.

    PENTING: fungsi ini TIDAK mengubah view_pan_x/y sehingga koordinat
    yang dihitung di mouse_callback selalu konsisten dengan tampilan.

    Algoritma:
      - Layar pixel (sx, sy) = logical pixel (view_pan_x + sx/zoom, view_pan_y + sy/zoom)
      - Logical pixel (lx, ly) tampil di screen (sx, sy) = ((lx-pan_x)*zoom, (ly-pan_y)*zoom)
    """
    lc_h, lc_w = logical_canvas.shape[:2]

    # Mulai dengan canvas hitam
    view = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

    # Region logical canvas yang terlihat di screen
    lx0 = view_pan_x
    ly0 = view_pan_y
    lx1 = view_pan_x + disp_w / view_zoom
    ly1 = view_pan_y + disp_h / view_zoom

    # Potong ke batas logical canvas
    src_x0 = max(0, int(lx0))
    src_y0 = max(0, int(ly0))
    src_x1 = min(lc_w, int(lx1) + 1)
    src_y1 = min(lc_h, int(ly1) + 1)

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        # Canvas di luar viewport sepenuhnya
        _draw_zoom_hint(view, disp_w, disp_h)
        return view

    # Posisi dst di screen (dimana crop ditempel)
    dst_x0 = max(0, int((src_x0 - lx0) * view_zoom))
    dst_y0 = max(0, int((src_y0 - ly0) * view_zoom))
    dst_x1 = min(disp_w, int((src_x1 - lx0) * view_zoom) + 1)
    dst_y1 = min(disp_h, int((src_y1 - ly0) * view_zoom) + 1)

    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        _draw_zoom_hint(view, disp_w, disp_h)
        return view

    crop   = logical_canvas[src_y0:src_y1, src_x0:src_x1]
    w_dst  = dst_x1 - dst_x0
    h_dst  = dst_y1 - dst_y0
    interp = cv2.INTER_NEAREST if view_zoom >= 4 else cv2.INTER_LINEAR

    try:
        resized = cv2.resize(crop, (w_dst, h_dst), interpolation=interp)
        view[dst_y0:dst_y1, dst_x0:dst_x1] = resized
    except Exception:
        pass

    _draw_zoom_hint(view, disp_w, disp_h)
    return view


def _draw_zoom_hint(view, disp_w, disp_h):
    """Overlay teks zoom + kontrol di pojok kiri-bawah layar."""
    hint = (f"  Zoom {view_zoom:.2f}x  |  Scroll=zoom  |  Klik-kanan+geser=pan"
            f"  |  F=fit  |  1=100%")
    (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    by = disp_h - 6
    cv2.rectangle(view, (0, by - hh - 4), (hw + 8, disp_h), (0, 0, 0), -1)
    cv2.putText(view, hint, (4, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 200, 100), 1, cv2.LINE_AA)


def mouse_callback(event, x, y, flags, param):
    """
    Koordinat (x,y) = piksel di window (screen space).
    Dikonversi: screen → logical canvas → image coords.

    Kontrol:
      Scroll          → zoom in/out centered di kursor
      Klik-kanan+geser→ pan viewport
      Klik-kiri dekat titik → drag titik
      Klik-kiri kosong      → tambah titik baru
      F (key)         → fit/reset zoom ke 1:1
    """
    global cal_pts, dragging_idx
    global view_zoom, view_pan_x, view_pan_y
    global panning, pan_start_s, pan_start_p
    global hover_idx
    padding = param

    # Konversi screen -> logical canvas -> image
    lx, ly  = screen_to_logical(x, y)
    img_x   = int(round(lx)) - padding
    img_y   = int(round(ly)) - padding

    # Helper: cek jarak ke semua titik (screen px)
    def nearest_pt():
        best_i, best_d = -1, float('inf')
        for i, pt in enumerate(cal_pts):
            pt_sx = (pt[0] + padding - view_pan_x) * view_zoom
            pt_sy = (pt[1] + padding - view_pan_y) * view_zoom
            d = ((x - pt_sx) ** 2 + (y - pt_sy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_i = d, i
        return (best_i if best_d <= DRAG_RADIUS else -1)

    # -- Scroll mouse: zoom --
    if event == cv2.EVENT_MOUSEWHEEL:
        factor   = ZOOM_STEP if flags > 0 else 1.0 / ZOOM_STEP
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, view_zoom * factor))
        view_pan_x = lx - x / new_zoom
        view_pan_y = ly - y / new_zoom
        view_zoom  = new_zoom
        return

    # -- Klik-kanan: mulai pan --
    if event == cv2.EVENT_RBUTTONDOWN:
        panning     = True
        pan_start_s = (x, y)
        pan_start_p = (view_pan_x, view_pan_y)
        return

    if event == cv2.EVENT_MOUSEMOVE and panning:
        dx = (x - pan_start_s[0]) / view_zoom
        dy = (y - pan_start_s[1]) / view_zoom
        view_pan_x = pan_start_p[0] - dx
        view_pan_y = pan_start_p[1] - dy
        return

    if event == cv2.EVENT_RBUTTONUP:
        panning = False
        return

    # -- Mouse move: update drag position atau hover highlight --
    if event == cv2.EVENT_MOUSEMOVE:
        if dragging_idx >= 0:
            cal_pts[dragging_idx] = (img_x, img_y)
        else:
            hover_idx = nearest_pt()   # highlight titik terdekat
        return

    # -- Klik-kiri down: mulai drag atau tambah titik --
    if event == cv2.EVENT_LBUTTONDOWN:
        hit = nearest_pt()
        if hit >= 0:
            dragging_idx = hit
            hover_idx    = -1
        elif len(cal_pts) < 4:
            cal_pts.append((img_x, img_y))
        return

    # -- Klik-kiri up: drop titik --
    if event == cv2.EVENT_LBUTTONUP:
        if dragging_idx >= 0:
            cal_pts[dragging_idx] = (img_x, img_y)
            dragging_idx = -1
            hover_idx    = nearest_pt()   # re-detect hover setelah drop


# ─── MAIN ─────────────────────────────────────────────────────────────────────

SRC_FIELDS = ["image", "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y",
              "real_width_mm", "real_depth_mm", "bev_w", "bev_h"]


def ask(prompt, default=""):
    try:
        ans = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    return ans or default


def write_src_pts(rows):
    with open(SRC_PTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SRC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n💾  src_pts -> {SRC_PTS_CSV.resolve()}  ({len(rows)} foto)")


def update_widths(rows, realwidth_map):
    """
    Mode [W]: samakan real_width_mm dengan folder 'real width/' dan hitung
    ulang bev_h = round(real_depth_mm x bev_w / real_width_mm).
    Titik acuan tidak berubah.
    """
    changes = []
    for r in rows:
        w_old, h_old = int(float(r["real_width_mm"])), int(float(r["bev_h"]))
        w_new = realwidth_map.get(r["image"])
        if w_new is None:
            print(f"   ⚠  {r['image']} tidak ada di folder 'real width/', dibiarkan.")
            w_new = w_old
        h_new = round(int(float(r["real_depth_mm"])) * int(float(r["bev_w"])) / w_new)
        if (w_new, h_new) != (w_old, h_old):
            changes.append((r["image"], w_old, h_old, w_new, h_new))
            r["real_width_mm"], r["bev_h"] = w_new, h_new
    if not changes:
        print("✅  Lebar jalan dan bev_h sudah sesuai. Tidak ada perubahan.")
        return
    print(f"\n   {len(changes)} foto akan berubah:")
    for img, w0, h0, w1, h1 in changes:
        print(f"   {img[:40]}: lebar {w0} -> {w1} mm, bev_h {h0} -> {h1} px")
    if ask("   Simpan ke src_pts.csv? (y/N): ", "n").lower() == "y":
        write_src_pts(rows)
    else:
        print("   Tidak disimpan.")


def main():
    global cal_pts, dragging_idx, hover_idx, _bev_last_pts
    global view_zoom, view_pan_x, view_pan_y, panning
    global REAL_WIDTH_MM, BEV_H, SCALE

    # Cari gambar di folder images/
    if not IMAGES_DIR.exists():
        print(f"[X]  Folder images tidak ditemukan: {IMAGES_DIR}")
        return
    image_files = sorted([
        f for f in IMAGES_DIR.iterdir()
        if f.suffix.lower() in IMG_EXTS
    ])

    if not image_files:
        print(f"[X]  Tidak ada gambar di: {BASE_DIR}")
        return

    # ── Bangun peta real_width per gambar dari folder 'real width/' ─────────
    realwidth_map = build_realwidth_map()
    mapped_count  = len(realwidth_map)

    print("=" * 60)
    print("  LANGKAH 4  -  KALIBRASI IPM (4 titik acuan)")
    print("=" * 60)
    print(f"[DIR]  Foto      : {IMAGES_DIR}")
    print(f"[ROAD] Kedalaman : {REAL_DEPTH_MM}mm (tetap untuk semua gambar)")
    print(f"[ROAD] Lebar     : per-gambar dari folder 'real width/' ({mapped_count} gambar terpetakan)")
    print(f"[BEV]  BEV lebar : {BEV_W}px tetap  |  tinggi dihitung proporsional per gambar")
    print(f"[IMG]  Gambar    : {len(image_files)} file ditemukan")
    print(f"[PAD]  Padding   : {PADDING}px | Display: {DISP_MAX_W}x{DISP_MAX_H}px")
    print("-" * 60)

    # ── Baca src_pts.csv yang sudah ada ──────────────────────────────────────
    existing_src_pts  = []
    images_calibrated = set()
    if SRC_PTS_CSV.exists():
        with open(SRC_PTS_CSV, "r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                existing_src_pts.append(dict(row))
                images_calibrated.add(row["image"])

    adjust_mode = False
    if images_calibrated:
        print(f"\n📋  src_pts.csv sudah berisi {len(images_calibrated)} foto.")
        print("   [N] Baru  - hanya foto yang belum dikalibrasi (default)")
        print("   [P] Pilih - atur ulang titik pada foto tertentu")
        print("   [A] Semua - atur ulang titik semua foto")
        print("   [W] Lebar - perbarui lebar jalan & bev_h dari folder 'real width/' (tanpa klik)")
        choice = ask("   Pilihan (N/P/A/W): ", "N").upper()

        if choice == "W":
            update_widths(existing_src_pts, realwidth_map)
            return
        if choice == "A":
            adjust_mode = True
            image_files = [f for f in image_files if f.name in images_calibrated] + \
                          [f for f in image_files if f.name not in images_calibrated]
        elif choice == "P":
            adjust_mode = True
            listed = [f for f in image_files if f.name in images_calibrated]
            for i, f in enumerate(listed, 1):
                print(f"   [{i:2d}] {f.name}")
            ans = ask("   Nomor foto (contoh: 3 atau 1,4,7): ")
            image_files = [listed[int(p) - 1] for p in ans.replace(" ", "").split(",")
                           if p.isdigit() and 1 <= int(p) <= len(listed)]
        else:
            image_files = [f for f in image_files if f.name not in images_calibrated]

    if not image_files:
        print("✅  Tidak ada foto yang perlu diproses.")
        return
    print(f"\U0001f5bc  Akan diproses: {len(image_files)} foto")
    print("-" * 60)

    all_src_pts = []
    all_meas    = []

    # Buat window dengan WINDOW_NORMAL agar bisa di-resize dan posisi dikontrol
    cv2.namedWindow(WINDOW_ORIG, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_BEV,  cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_ORIG, mouse_callback, PADDING)

    for img_idx, img_path in enumerate(image_files):
        cal_pts      = []
        dragging_idx = -1   # reset drag state per gambar (via global)
        hover_idx    = -1   # reset hover

        # Mode adjust: pre-load titik lama dari src_pts.csv
        if adjust_mode and img_path.name in images_calibrated:
            for row in existing_src_pts:
                if row["image"] == img_path.name:
                    try:
                        cal_pts = [
                            (int(float(row["tl_x"])), int(float(row["tl_y"]))),
                            (int(float(row["tr_x"])), int(float(row["tr_y"]))),
                            (int(float(row["br_x"])), int(float(row["br_y"]))),
                            (int(float(row["bl_x"])), int(float(row["bl_y"]))),
                        ]
                        print(f"   \U0001f527  Titik lama dimuat — drag untuk sesuaikan, C/Enter untuk simpan")
                    except Exception:
                        cal_pts = []
                        print(f"   \u26a0  Gagal load titik lama, mulai dari awal")
                    break

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠  Gagal membuka: {img_path.name}")
            continue

        img_h, img_w = img.shape[:2]

        # ── Real width per gambar ─────────────────────────────────────────────
        img_real_width = realwidth_map.get(img_path.name, DEFAULT_WIDTH_MM)
        if img_path.name not in realwidth_map:
            print(f"   ⚠  '{img_path.name}' tidak ada di folder 'real width/' → pakai DEFAULT {DEFAULT_WIDTH_MM}mm")

        # BEV height & scale dihitung ulang per gambar sesuai real_width-nya
        img_bev_h = round(REAL_DEPTH_MM * BEV_W / img_real_width)
        img_scale = img_real_width / BEV_W

        # Update globals agar draw_orig & draw_bev pakai nilai yang benar per-gambar
        REAL_WIDTH_MM = img_real_width
        BEV_H         = img_bev_h
        SCALE         = img_scale

        # Ukuran logical canvas = gambar + padding semua sisi
        canvas_w = img_w + 2 * PADDING
        canvas_h = img_h + 2 * PADDING

        # Ukuran display window (dibatasi DISP_MAX agar tidak meluber layar)
        disp_w = min(canvas_w, DISP_MAX_W)
        disp_h = min(canvas_h, DISP_MAX_H)
        cv2.resizeWindow(WINDOW_ORIG, disp_w, disp_h)

        # Reset zoom/pan per gambar: tampilkan seluruh canvas fit di window
        view_zoom  = min(disp_w / canvas_w, disp_h / canvas_h)
        view_pan_x = 0.0
        view_pan_y = 0.0
        panning    = False

        # Cari label YOLO
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            label_path = LABELS_DIR / (img_path.stem + ".txt")

        boxes = load_yolo_labels(label_path)
        print(f"\n🖼  [{img_idx+1}/{len(image_files)}] {img_path.name}")
        print(f"   Ukuran gambar : {img_w}x{img_h}px  |  Canvas: {canvas_w}x{canvas_h}px")
        print(f"   Real width    : {img_real_width}mm  |  Depth: {REAL_DEPTH_MM}mm  |  BEV: {BEV_W}x{img_bev_h}px  |  Scale: {img_scale:.2f}mm/px")
        print(f"   Label         : {len(boxes)} pothole")
        if not label_path.exists():
            print(f"   ⚠  Label tidak ditemukan: {label_path.name}")
        print(f"   💡  Titik boleh diklik di luar gambar (area abu-abu PADDING={PADDING}px)")

        action        = "continue"
        M             = None
        _bev_last_pts = None   # reset cache BEV setiap gambar baru

        while True:
            cur_M    = None    # homografi frame ini
            cur_meas = []      # measurement frame ini
            next_lbl = PT_LABELS[len(cal_pts)] if len(cal_pts) < 4 else "--"
            # Bangun logical canvas (full resolution)
            vis_orig = draw_orig(img, cal_pts, next_lbl, PADDING, dragging_idx, hover_idx)
            # Render ke display window (dengan zoom/pan)
            view = render_view(vis_orig, disp_w, disp_h)

            # ── BEV LIVE: window terpisah, hanya render saat titik berubah ────
            pts_key = tuple(map(tuple, cal_pts)) if len(cal_pts) == 4 else None
            if len(cal_pts) == 4 and pts_key != _bev_last_pts:
                _bev_last_pts = pts_key
                try:
                    cur_M = build_homography(cal_pts, BEV_W, BEV_H)
                    bev_live, cur_meas = draw_bev(cur_M, img, boxes, img_w, img_h)
                    # Scale BEV ke ukuran nyaman: lebar max 480px
                    bh_s, bw_s = bev_live.shape[:2]
                    dw = min(480, bw_s)
                    dh = int(bh_s * dw / bw_s) if bw_s > 0 else bh_s
                    bev_disp = cv2.resize(bev_live, (dw, dh), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow(WINDOW_BEV, bev_disp)
                    cv2.resizeWindow(WINDOW_BEV, dw, dh)
                except Exception:
                    pass
            elif len(cal_pts) < 4 and _bev_last_pts is not None:
                _bev_last_pts = None
                try: cv2.destroyWindow(WINDOW_BEV)
                except Exception: pass

            # Tampilkan gambar asli (ukuran window tidak berubah)
            cv2.imshow(WINDOW_ORIG, view)
            key = cv2.waitKeyEx(30)

            # Window ditutup paksa
            if cv2.getWindowProperty(WINDOW_ORIG, cv2.WND_PROP_VISIBLE) < 1:
                action = "quit"
                break

            if key in (ord('z'), ord('Z')):          # Z: undo titik terakhir
                if cal_pts:
                    removed = cal_pts.pop()
                    print(f"   Undo titik {len(cal_pts)+1} ({removed[0]}, {removed[1]}) -- tersisa {len(cal_pts)} titik")

            elif key in (ord('r'), ord('R')):         # R: reset
                cal_pts = []
                print("   Reset semua titik")

            elif key in (ord('p'), ord('P')):         # P: cetak measurement frame ini ke console
                if cur_meas:
                    print(f"   \U0001f4cf  Measurement saat ini ({len([m for m in cur_meas if m])} pothole):")
                    for m in cur_meas:
                        if m:
                            rel = {"OK": "\u2705", "~": "\u26a0", "!": "\u274c"}.get(m['reliability'], "")
                            print(f"      #{m['pothole_idx']} {rel}: "
                                  f"{m['w_mm']:.1f} x {m['h_mm']:.1f} mm  "
                                  f"\u2192 {m['area_cm2']:.2f} cm\u00b2  "
                                  f"(trap={m['trapezoid_ratio']:.2f})")
                else:
                    if len(cal_pts) < 4:
                        print(f"   Perlu 4 titik dulu! Baru ada {len(cal_pts)} titik.")
                    else:
                        print("   Tidak ada pothole dalam area BEV.")

            elif key in (ord('f'), ord('F')):         # F: fit zoom ke seluruh canvas
                view_zoom  = min(disp_w / canvas_w, disp_h / canvas_h)
                view_pan_x = 0.0
                view_pan_y = 0.0

            elif key in (ord('1'),):                  # 1: zoom 1:1 (piksel asli)
                view_zoom  = 1.0
                view_pan_x = PADDING - disp_w  * 0.5 / view_zoom
                view_pan_y = PADDING - disp_h  * 0.5 / view_zoom

            elif key in (ord('c'), ord('C'), 13):     # C/Enter: konfirmasi & simpan
                if len(cal_pts) < 4:
                    print(f"   Perlu 4 titik! Baru ada {len(cal_pts)} titik.")
                else:
                    # Pakai M & meas dari frame terakhir (sudah dihitung di atas)
                    M            = cur_M
                    measurements = cur_meas
                    if M is None:
                        # Fallback: hitung ulang jika frame pertama belum sempat render
                        try:
                            M = build_homography(cal_pts, BEV_W, BEV_H)
                            _, measurements = draw_bev(M, img, boxes, img_w, img_h)
                        except Exception as e:
                            print(f"   [!] Gagal hitung BEV: {e}")
                            M, measurements = None, []
                    action = "confirmed"
                    break

            elif key in (ord('s'), ord('S')):         # S: skip
                action = "skip"
                break

            elif key in (ord('q'), ord('Q'), 27):     # Q/Esc: keluar
                action = "quit"
                break

        # ── Simpan hasil ──────────────────────────────────────────────────────
        if action == "confirmed" and M is not None:
            tl, tr, br, bl = cal_pts

            # Simpan src_pts (koordinat dalam sistem gambar, bisa negatif)
            all_src_pts.append({
                "image":         img_path.name,
                "tl_x": tl[0],  "tl_y": tl[1],
                "tr_x": tr[0],  "tr_y": tr[1],
                "br_x": br[0],  "br_y": br[1],
                "bl_x": bl[0],  "bl_y": bl[1],
                "real_width_mm": img_real_width,   # per-gambar dari folder real width/
                "real_depth_mm": REAL_DEPTH_MM,    # 5000 untuk semua gambar
                "bev_w":         BEV_W,
                "bev_h":         img_bev_h,        # proporsional per gambar
            })

            print(f"   ✅  Disimpan — {len([m for m in measurements if m])} pothole terukur:")
            for m in measurements:
                if m is None:
                    print(f"      ⚠  Pothole di luar area BEV — skip")
                    continue
                all_meas.append({
                    "image":           img_path.name,
                    "pothole_idx":     m["pothole_idx"],
                    "w_mm":            m["w_mm"],
                    "h_mm":            m["h_mm"],
                    "area_mm2":        m["area_mm2"],
                    "w_cm":            m["w_cm"],
                    "h_cm":            m["h_cm"],
                    "area_cm2":        m["area_cm2"],
                    "trapezoid_ratio": m["trapezoid_ratio"],
                    "reliability":     m["reliability"],
                    "bev_w":           BEV_W,
                    "bev_h":           img_bev_h,
                    "scale_mm_px":     round(img_scale, 4),
                    "real_width_mm":   img_real_width,
                    "real_depth_mm":   REAL_DEPTH_MM,
                })
                rel_icon = {"OK": "✅", "~": "⚠", "!": "❌"}.get(m['reliability'], "")
                print(f"      #{m['pothole_idx']} {rel_icon}: "
                      f"{m['w_mm']:.1f} x {m['h_mm']:.1f} mm  "
                      f"→  {m['area_cm2']:.2f} cm²  "
                      f"(trapezoid ratio: {m['trapezoid_ratio']:.2f})")

        elif action == "skip":
            print("   ⏭  Di-skip")

        elif action == "quit":
            print("\n🚪  Keluar...")
            break

    cv2.destroyAllWindows()

    # ── Tulis CSV ─────────────────────────────────────────────────────────────
    if not all_src_pts:
        print("\n⚠  Tidak ada data baru. src_pts.csv tidak diubah.")
        return

    updated_names = {r["image"] for r in all_src_pts}
    new_by_name   = {r["image"]: r for r in all_src_pts}
    combined_src  = [new_by_name.get(r["image"], r) for r in existing_src_pts] + \
                    [r for r in all_src_pts if r["image"] not in images_calibrated]
    n_updated     = len(updated_names & images_calibrated)
    n_new         = len(all_src_pts) - n_updated
    print("\n" + "-" * 60)
    print(f"   {n_updated} foto diperbarui, {n_new} foto baru. Total: {len(combined_src)} foto")
    if ask("   Simpan ke src_pts.csv dan ipm_measurements.csv? (y/N): ", "n").lower() != "y":
        print("   Tidak disimpan. src_pts.csv tetap seperti semula.")
        return

    write_src_pts(combined_src)

    # ipm_measurements.csv: baris lama untuk foto yang diproses ulang diganti
    old_meas = []
    if MEAS_CSV.exists():
        with open(MEAS_CSV, "r", newline="", encoding="utf-8-sig") as f:
            old_meas = [dict(r) for r in csv.DictReader(f)]
    combined_meas = [r for r in old_meas if r["image"] not in updated_names] + all_meas
    if combined_meas:
        with open(MEAS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_meas[0].keys()) if all_meas
                                    else list(combined_meas[0].keys()))
            writer.writeheader()
            writer.writerows(combined_meas)
        print(f"💾  measurements -> {MEAS_CSV.resolve()}  ({len(combined_meas)} baris)")


if __name__ == "__main__":
    main()
