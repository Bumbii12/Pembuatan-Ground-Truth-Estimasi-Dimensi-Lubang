"""
1_build_pothole_gt.py
=====================
LANGKAH 1. Mengukur ukuran asli setiap lubang dari foto tampak atas.

Setiap foto di top_down/images memuat satu lubang dan selembar kertas
setengah A4 (14,85 x 21,0 cm) sebagai objek referensi. Pengguna menggambar
kotak di atas kertas itu, lalu skrip menghitung piksel per cm:

    px_per_cm   = lebar_kotak_referensi_px / 14,85
    lebar_cm    = lebar_bbox_lubang_px   / px_per_cm
    panjang_cm  = tinggi_bbox_lubang_px  / px_per_cm

Hasilnya disimpan ke pothole_gt.csv dengan kode tetap HOLE_001, HOLE_002, ...

Struktur folder:
  Pembuatan_GT/
  ├── top_down/
  │   ├── images/      <- foto lubang tampak atas (dengan kertas setengah A4)
  │   └── labels/      <- bbox lubang, format YOLO: class cx cy w h
  ├── pothole_gt.csv   <- OUTPUT
  └── 1_build_pothole_gt.py

Mode saat pothole_gt.csv sudah ada:
  [L] Lanjut     : hanya memproses foto yang belum tercatat (default)
  [U] Ukur ulang : pilih foto tertentu untuk diukur ulang. Kode HOLE tetap,
                   hanya ukurannya yang diperbarui.
  Sebelum menulis file, skrip selalu meminta konfirmasi (y/N), jadi demo
  bisa dijalankan tanpa mengubah data.

Kontrol:
  Drag klik-kiri  -> Gambar kotak referensi (rasio setengah A4 terkunci)
  Scroll mouse    -> Perbesar / perkecil kotak referensi
  <- -> / A D     -> Putar kotak 1 derajat
  Klik-kanan drag -> Geser gambar
  C / Enter       -> Konfirmasi ukuran lubang di gambar ini & lanjut
  R               -> Reset kotak referensi
  S               -> Skip gambar ini
  Q / Esc         -> Keluar
"""

import cv2
import numpy as np
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─── BACA GAMBAR DENGAN EXIF ROTATION ────────────────────────────────────────

def imread_respecting_exif(path):
    """
    Baca gambar dan terapkan rotasi EXIF jika ada.
    Tanpa ini, JPEG dari kamera bisa tampil miring/terbalik di OpenCV.
    Membutuhkan Pillow (pip install Pillow).
    """
    try:
        from PIL import Image, ImageOps
        pil_img = Image.open(str(path))
        pil_img = ImageOps.exif_transpose(pil_img)  # terapkan rotasi EXIF
        img_rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except ImportError:
        print("⚠  Pillow tidak tersedia, gambar mungkin tampil miring.")
        print("   Pasang dengan: pip install Pillow")
        return cv2.imread(str(path))
    except Exception:
        # Fallback ke cv2 jika ada error lain
        return cv2.imread(str(path))

# ─── KONFIGURASI ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
TOPDOWN_DIR    = BASE_DIR / "top_down"
IMAGES_DIR     = TOPDOWN_DIR / "images"
LABELS_DIR     = TOPDOWN_DIR / "labels"
OUTPUT_CSV     = BASE_DIR / "pothole_gt.csv"
VALIDATION_CSV = BASE_DIR / "dataset_final" / "validation_gt.csv"

# Objek referensi: setengah kertas A4 (A5), posisi potret
REF_W_CM     = 14.85       # lebar (cm)
REF_H_CM     = 21.0        # tinggi (cm)
ASPECT_RATIO = REF_W_CM / REF_H_CM   # = 0.7071

FIELDNAMES = [
    "pothole_id", "gt_image", "gt_label_idx",
    "yolo_cx", "yolo_cy", "yolo_w", "yolo_h",
    "gt_w_cm", "gt_h_cm", "gt_area_cm2",
]

# ─── LAYOUT WINDOW ────────────────────────────────────────────────────────────
WINDOW_NAME = "Build Pothole GT  —  ½ A4 Object Reference"
HEADER_H    = 60
FOOTER_H    = 50
CANVAS_W    = 1100
IMG_AREA_H  = 750
CANVAS_H    = IMG_AREA_H + HEADER_H + FOOTER_H

# ─── STATE GLOBAL ─────────────────────────────────────────────────────────────
state = {
    "drawing":     False,
    "start":       None,
    "ref_corners": None,
    "ref_center":  None,
    "ref_angle":   0.0,
    "ref_w_px":    0.0,
    "img_w":       0,
    "img_h":       0,
    "zoom":        1.0,
    "pan_x":       0,
    "pan_y":       0,
    "panning":     False,
    "pan_start":   (0, 0),
}


# ─── KOORDINAT HELPER ─────────────────────────────────────────────────────────

def img_to_screen(x, y):
    sx = int(x * state["zoom"] + state["pan_x"])
    sy = int(y * state["zoom"] + state["pan_y"]) + HEADER_H
    return sx, sy


def screen_to_img(sx, sy):
    x = (sx - state["pan_x"]) / state["zoom"]
    y = (sy - HEADER_H - state["pan_y"]) / state["zoom"]
    return x, y


# ─── GEOMETRI HELPER ──────────────────────────────────────────────────────────

def compute_corners(cx, cy, w_px, angle_deg):
    """Hitung 4 sudut rotated rectangle dari pusat, lebar, dan sudut."""
    h_px = w_px / ASPECT_RATIO
    hw, hh = w_px / 2, h_px / 2
    rad = np.deg2rad(angle_deg)
    ca, sa = np.cos(rad), np.sin(rad)
    rel = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    corners = []
    for rx, ry in rel:
        x = cx + rx * ca - ry * sa
        y = cy + rx * sa + ry * ca
        corners.append((int(round(x)), int(round(y))))
    return corners


def corners_from_drag(start, end):
    """
    Bangun rotated rect dari drag:
      start → satu ujung edge lebar (P0)
      end   → ujung lain edge lebar (P1)
    Tinggi dihitung otomatis dari aspek rasio.
    """
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    dist = np.hypot(dx, dy)
    if dist < 5:
        return None, None, None, None

    w_px = dist
    h_px = w_px / ASPECT_RATIO

    ux, uy = dx / dist, dy / dist   # unit vector arah drag (lebar)
    px, py = -uy, ux                 # tegak lurus (tinggi)

    P0 = (ax,                 ay)
    P1 = (ax + ux * w_px,     ay + uy * w_px)
    P2 = (P1[0] + px * h_px,  P1[1] + py * h_px)
    P3 = (P0[0] + px * h_px,  P0[1] + py * h_px)

    corners = [(int(round(p[0])), int(round(p[1]))) for p in [P0, P1, P2, P3]]
    center  = ((P0[0] + P2[0]) / 2, (P0[1] + P2[1]) / 2)
    angle   = float(np.degrees(np.arctan2(uy, ux)))

    return corners, center, angle, w_px


# ─── MOUSE CALLBACK ───────────────────────────────────────────────────────────

def mouse_callback(event, x, y, flags, param):
    ix, iy = screen_to_img(x, y)
    iw, ih = state["img_w"], state["img_h"]
    ix_c = int(max(0, min(ix, iw - 1)))
    iy_c = int(max(0, min(iy, ih - 1)))

    if event == cv2.EVENT_LBUTTONDOWN:
        if y < HEADER_H or y > CANVAS_H - FOOTER_H:
            return
        state["drawing"]     = True
        state["start"]       = (ix_c, iy_c)
        state["ref_corners"] = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if state["drawing"]:
            corners, center, angle, w_px = corners_from_drag(state["start"], (ix_c, iy_c))
            if corners:
                state["ref_corners"] = corners
                state["ref_center"]  = center
                state["ref_angle"]   = angle
                state["ref_w_px"]    = w_px
        elif state["panning"]:
            dx = x - state["pan_start"][0]
            dy = y - state["pan_start"][1]
            state["pan_x"] += dx
            state["pan_y"] += dy
            state["pan_start"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        state["drawing"] = False
        corners, center, angle, w_px = corners_from_drag(state["start"], (ix_c, iy_c))
        if corners and w_px > 10:
            state["ref_corners"] = corners
            state["ref_center"]  = center
            state["ref_angle"]   = angle
            state["ref_w_px"]    = w_px
        else:
            state["ref_corners"] = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        state["panning"]   = True
        state["pan_start"] = (x, y)

    elif event == cv2.EVENT_RBUTTONUP:
        state["panning"] = False

    elif event == cv2.EVENT_MOUSEWHEEL and state["ref_corners"] is not None:
        factor = 1.05 if (flags > 0) else 0.95
        new_w  = max(20.0, state["ref_w_px"] * factor)
        cx, cy = state["ref_center"]
        state["ref_w_px"]    = new_w
        state["ref_corners"] = compute_corners(cx, cy, new_w, state["ref_angle"])


def rotate_ref(delta_deg):
    """Putar kotak referensi sebesar delta_deg derajat."""
    if state["ref_corners"] is None:
        return
    state["ref_angle"] += delta_deg
    cx, cy = state["ref_center"]
    state["ref_corners"] = compute_corners(cx, cy, state["ref_w_px"], state["ref_angle"])


# ─── RENDER ───────────────────────────────────────────────────────────────────

def draw_scene(base_img, pothole_boxes, ref_corners, img_w, img_h,
               idx, total, img_name, measurements=None):
    zoom  = state["zoom"]
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    area_y1, area_y2 = HEADER_H, CANVAS_H - FOOTER_H

    # ── Render gambar ──
    new_w = max(1, int(img_w * zoom))
    new_h = max(1, int(img_h * zoom))
    zoomed = cv2.resize(base_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ox = int(state["pan_x"]);  oy = int(state["pan_y"]) + HEADER_H
    x1_c = max(0, ox);              y1_c = max(area_y1, oy)
    x2_c = min(CANVAS_W, ox+new_w); y2_c = min(area_y2, oy+new_h)
    x1_z = max(0, -ox);             y1_z = max(0, area_y1 - oy)
    x2_z = x1_z + (x2_c - x1_c);   y2_z = y1_z + (y2_c - y1_c)
    if x2_c > x1_c and y2_c > y1_c and x2_z <= new_w and y2_z <= new_h:
        canvas[y1_c:y2_c, x1_c:x2_c] = zoomed[y1_z:y2_z, x1_z:x2_z]

    # ── Pothole bounding boxes ──
    for i, (cx, cy, bw, bh) in enumerate(pothole_boxes):
        sx1, sy1 = img_to_screen((cx - bw/2) * img_w, (cy - bh/2) * img_h)
        sx2, sy2 = img_to_screen((cx + bw/2) * img_w, (cy + bh/2) * img_h)
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (0, 60, 220), 2)
        lbl = f"Lubang #{i+1}"
        if measurements and i < len(measurements):
            m = measurements[i]
            lbl += f"  {m['gt_w_cm']:.1f} x {m['gt_h_cm']:.1f} cm"
        cv2.putText(canvas, lbl, (sx1, max(sy1 - 8, area_y1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 120, 255), 2)

    # ── Reference box ──
    if ref_corners is not None:
        sc  = [img_to_screen(c[0], c[1]) for c in ref_corners]
        pts = np.array(sc, dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 215, 0), thickness=2)
        cv2.arrowedLine(canvas, sc[0], sc[1], (0, 255, 80), 1, tipLength=0.15)
        scx, scy = img_to_screen(state["ref_center"][0], state["ref_center"][1])
        cv2.drawMarker(canvas, (scx, scy), (0, 215, 0), cv2.MARKER_CROSS, 10, 1)
        lbl_ref = f"REF 1/2 A4  {REF_W_CM}x{REF_H_CM}cm  [{state['ref_angle']:.1f} deg]"
        lx = min(c[0] for c in sc);  ly = min(c[1] for c in sc)
        cv2.putText(canvas, lbl_ref, (max(lx, 4), max(ly - 8, area_y1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 0), 2)

    # ── Header ──
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, HEADER_H), (18, 18, 30), -1)
    cv2.putText(canvas, f"[{idx+1}/{total}]  {img_name[:80]}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1)
    cv2.putText(canvas,
                f"Zoom: {zoom:.1f}x  |  Ref: {REF_W_CM} x {REF_H_CM} cm (1/2 A4)  |  "
                f"{len(pothole_boxes)} lubang terdeteksi",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 140), 1)

    # ── Footer ──
    cv2.rectangle(canvas, (0, CANVAS_H - FOOTER_H), (CANVAS_W, CANVAS_H), (18, 18, 30), -1)
    if ref_corners:
        status      = "REF SIAP  -  Tekan C / Enter untuk konfirmasi semua lubang"
        status_col  = (50, 255, 120)
    else:
        status      = "Drag klik-kiri untuk gambar kotak referensi di atas kertas 1/2 A4"
        status_col  = (50, 170, 255)
    cv2.putText(canvas, status, (8, CANVAS_H - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_col, 1)
    cv2.putText(canvas,
                "Scroll: resize ref  |  Klik-kanan drag: geser  |  <-->/AD: rotasi  |  "
                "R: reset  |  S: skip gambar  |  Q: keluar & simpan",
                (8, CANVAS_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 120, 120), 1)

    return canvas


# ─── KALKULASI ────────────────────────────────────────────────────────────────

def compute_measurements(pothole_boxes, ref_w_px, img_w, img_h):
    """Hitung dimensi fisik setiap lubang relatif terhadap object reference."""
    px_per_cm = ref_w_px / REF_W_CM
    results   = []
    for cx, cy, bw, bh in pothole_boxes:
        pot_w_px  = bw * img_w
        pot_h_px  = bh * img_h
        pot_w_cm  = pot_w_px / px_per_cm
        pot_h_cm  = pot_h_px / px_per_cm
        area_cm2  = pot_w_cm * pot_h_cm
        results.append({
            "gt_w_cm":     round(pot_w_cm, 2),
            "gt_h_cm":     round(pot_h_cm, 2),
            "gt_area_cm2": round(area_cm2, 2),
        })
    return results


def load_yolo_labels(label_path):
    """Baca file label YOLO .txt -> list [cx, cy, bw, bh]."""
    boxes = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append([float(p) for p in parts[1:5]])
    return boxes


def hole_num(hole_id):
    """HOLE_012 -> 12"""
    try:
        return int(str(hole_id).split("_")[-1])
    except ValueError:
        return 0


def load_existing_rows():
    """Baca pothole_gt.csv yang ada -> list of dict (urutan sesuai file)."""
    if not OUTPUT_CSV.exists():
        return []
    with open(OUTPUT_CSV, "r", newline="", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def ask(prompt, default=""):
    try:
        ans = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    return ans or default


def write_pothole_csv(rows):
    rows = sorted(rows, key=lambda r: hole_num(r["pothole_id"]))
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅  Disimpan ke: {OUTPUT_CSV.resolve()}  ({len(rows)} lubang)")


def sync_validation(rows):
    """
    Salin ukuran terbaru dari pothole_gt.csv ke validation_gt.csv
    (kolom gt_w_cm, gt_h_cm, gt_area_cm2), dicocokkan lewat pothole_id.
    """
    if not VALIDATION_CSV.exists():
        print(f"   ⚠  {VALIDATION_CSV} belum ada, sinkronisasi dilewati.")
        return
    lookup = {r["pothole_id"]: r for r in rows}
    with open(VALIDATION_CSV, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        vrows  = [dict(r) for r in reader]
    n_changed = 0
    for v in vrows:
        h = lookup.get(v["pothole_id"])
        if h is None:
            continue
        for k in ("gt_w_cm", "gt_h_cm", "gt_area_cm2"):
            if str(v[k]) != str(h[k]):
                v[k] = h[k]
                n_changed += 1
    with open(VALIDATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(vrows)
    print(f"✅  validation_gt.csv disinkronkan ({n_changed} nilai berubah)")


def choose_images(image_files, existing_rows):
    """Tampilkan daftar foto yang sudah tercatat, lalu minta nomor yang mau diukur ulang."""
    by_img = {}
    for r in existing_rows:
        by_img.setdefault(r["gt_image"], []).append(r["pothole_id"])
    listed = [f for f in image_files if f.name in by_img]
    print("\n   Foto yang sudah tercatat:")
    for i, f in enumerate(listed, 1):
        print(f"   [{i:2d}] {', '.join(by_img[f.name])}  -  {f.name[:45]}")
    ans = ask("   Nomor foto (contoh: 3 atau 1,4,7 atau 'semua'): ")
    if ans.lower() == "semua":
        return listed
    chosen = []
    for part in ans.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= len(listed):
            chosen.append(listed[int(part) - 1])
    return chosen


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if not IMAGES_DIR.exists():
        print(f"❌  Folder tidak ditemukan: {IMAGES_DIR.resolve()}")
        sys.exit(1)

    image_files = sorted(
        list(IMAGES_DIR.glob("*.jpg"))  +
        list(IMAGES_DIR.glob("*.jpeg")) +
        list(IMAGES_DIR.glob("*.png"))
    )
    if not image_files:
        print(f"❌  Tidak ada gambar di: {IMAGES_DIR.resolve()}")
        sys.exit(1)

    existing_rows = load_existing_rows()
    images_done   = {r["gt_image"] for r in existing_rows}
    next_hole     = max([hole_num(r["pothole_id"]) for r in existing_rows], default=0)

    print("=" * 60)
    print("  LANGKAH 1  -  BUILD POTHOLE GT (objek referensi setengah A4)")
    print("=" * 60)
    print(f"📂  Foto tampak atas : {IMAGES_DIR.resolve()}")
    print(f"💾  Output           : {OUTPUT_CSV.resolve()}")
    print(f"📐  Referensi        : {REF_W_CM} x {REF_H_CM} cm  (rasio {ASPECT_RATIO:.4f})")

    mode = "L"
    if existing_rows:
        print(f"\n📋  pothole_gt.csv sudah berisi {len(existing_rows)} lubang dari {len(images_done)} foto.")
        print("   [L] Lanjut     - hanya foto yang belum tercatat (default)")
        print("   [U] Ukur ulang - pilih foto tertentu, kode HOLE tetap")
        mode = ask("   Pilihan (L/U): ", "L").upper()

    if mode == "U":
        image_files = choose_images(image_files, existing_rows)
    else:
        image_files = [f for f in image_files if f.name not in images_done]

    print(f"🖼  Akan diproses: {len(image_files)} foto")
    print("─" * 60)
    if not image_files:
        print("✅  Tidak ada foto yang perlu diproses.")
        return

    # Hasil baru per foto: {gt_image: [row, ...]}
    new_by_img = {}

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    for idx, img_path in enumerate(image_files):
        state.update({
            "ref_corners": None, "ref_center": None,
            "ref_angle":   0.0,  "ref_w_px":   0.0,
            "drawing":     False, "panning":    False,
        })

        base_img = imread_respecting_exif(img_path)
        if base_img is None:
            print(f"⚠  Gagal membuka: {img_path.name}")
            continue

        state["img_h"], state["img_w"] = base_img.shape[:2]
        img_w, img_h = state["img_w"], state["img_h"]

        # Auto-fit zoom
        fit_zoom = min(CANVAS_W / img_w, IMG_AREA_H / img_h)
        state["zoom"]  = fit_zoom
        state["pan_x"] = int((CANVAS_W - img_w * fit_zoom) / 2)
        state["pan_y"] = int((IMG_AREA_H - img_h * fit_zoom) / 2)

        label_path    = LABELS_DIR / (img_path.stem + ".txt")
        pothole_boxes = load_yolo_labels(label_path)

        if not pothole_boxes:
            print(f"⚠  [{idx+1}/{len(image_files)}] Tidak ada label YOLO: {img_path.name} - skip")
            continue

        print(f"\n🖼  [{idx+1}/{len(image_files)}] {img_path.name}  |  {len(pothole_boxes)} lubang")
        old_rows = {int(r["gt_label_idx"]): r for r in existing_rows if r["gt_image"] == img_path.name}
        for li, r in sorted(old_rows.items()):
            print(f"   Nilai lama {r['pothole_id']}: {r['gt_w_cm']} x {r['gt_h_cm']} cm")

        measurements = None
        action = "continue"

        while True:
            if state["ref_corners"] is not None:
                measurements = compute_measurements(
                    pothole_boxes, state["ref_w_px"], img_w, img_h)
            else:
                measurements = None

            canvas = draw_scene(
                base_img, pothole_boxes, state["ref_corners"],
                img_w, img_h, idx, len(image_files), img_path.name, measurements
            )
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKeyEx(30)

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                action = "quit"
                break

            if key in (2424832, ord('a'), ord('A')):       # <- / A : putar kiri
                rotate_ref(-1.0)
            elif key in (2555904, ord('d'), ord('D')):     # -> / D : putar kanan
                rotate_ref(+1.0)
            elif key in (ord('c'), ord('C'), 13):          # C / Enter : konfirmasi
                if state["ref_corners"] is None:
                    print("   ⚠  Gambar dulu kotak referensi di atas kertas setengah A4!")
                else:
                    action = "confirmed"
                    break
            elif key in (ord('r'), ord('R')):              # R : reset
                state["ref_corners"] = None
                print("   🔄  Referensi di-reset")
            elif key in (ord('s'), ord('S')):              # S : skip
                action = "skip"
                break
            elif key in (ord('q'), ord('Q'), 27):          # Q / Esc : keluar
                action = "quit"
                break

        if action == "confirmed" and measurements:
            rows = []
            for i, (box, m) in enumerate(zip(pothole_boxes, measurements)):
                if i in old_rows:
                    hole_id = old_rows[i]["pothole_id"]      # kode lama dipertahankan
                else:
                    next_hole += 1
                    hole_id = f"HOLE_{next_hole:03d}"
                rows.append({
                    "pothole_id":   hole_id,
                    "gt_image":     img_path.name,
                    "gt_label_idx": i,
                    "yolo_cx":      round(box[0], 6),
                    "yolo_cy":      round(box[1], 6),
                    "yolo_w":       round(box[2], 6),
                    "yolo_h":       round(box[3], 6),
                    "gt_w_cm":      m["gt_w_cm"],
                    "gt_h_cm":      m["gt_h_cm"],
                    "gt_area_cm2":  m["gt_area_cm2"],
                })
                print(f"   ✅  {hole_id}: {m['gt_w_cm']:.2f} x {m['gt_h_cm']:.2f} cm "
                      f"= {m['gt_area_cm2']:.2f} cm2")
            new_by_img[img_path.name] = rows
        elif action == "skip":
            print("   ⏭  Di-skip")
        elif action == "quit":
            print("\n🚪  Keluar...")
            break

    cv2.destroyAllWindows()

    if not new_by_img:
        print("\n⚠  Tidak ada data baru. pothole_gt.csv tidak diubah.")
        return

    # Gabungkan: baris lama untuk foto yang diproses ulang diganti
    combined = [r for r in existing_rows if r["gt_image"] not in new_by_img]
    for rows in new_by_img.values():
        combined.extend(rows)

    print("\n" + "─" * 60)
    print(f"   {len(new_by_img)} foto diukur. Total lubang setelah disimpan: {len(combined)}")
    if ask("   Simpan ke pothole_gt.csv? (y/N): ", "n").lower() != "y":
        print("   Tidak disimpan. pothole_gt.csv tetap seperti semula.")
        return
    write_pothole_csv(combined)

    if mode == "U" and VALIDATION_CSV.exists():
        if ask("   Perbarui juga ukuran di dataset_final/validation_gt.csv? (y/N): ", "n").lower() == "y":
            sync_validation(combined)

    print("\nSelesai!")


if __name__ == "__main__":
    main()
