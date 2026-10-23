"""
3_build_validation_gt.py
========================
LANGKAH 3. Memetakan setiap lubang di foto tampak depan (POV kendaraan)
ke kode HOLE dari langkah 1.

Untuk setiap foto di dataset_final/pov_kendaraan/images, skrip menampilkan
bbox dari labels/. Pengguna memilih lubang mana itu dengan bantuan thumbnail
foto tampak atas. Ukuran asli lubang lalu disalin dari pothole_gt.csv.
Hasilnya dataset_final/validation_gt.csv (REAL_001 ... REAL_100).

Tampilan:
  Panel KIRI  : foto POV kendaraan dengan bbox YOLO
  Panel KANAN : thumbnail foto tampak atas (top_down/images) sebagai acuan

Mode saat validation_gt.csv sudah ada:
  [L] Lanjut  : hanya anotasi yang belum dipetakan (default)
  [U] Ubah    : pilih foto tertentu dan ubah pemetaannya. REAL_ID tetap.
  [S] Sinkron : salin ulang ukuran dari pothole_gt.csv, tanpa membuka jendela
  Sebelum menulis file, skrip selalu meminta konfirmasi (y/N).

Kontrol:
  1 ... 9         -> Pilih HOLE untuk lubang aktif (kuning)
  Klik thumbnail  -> Pilih HOLE dari panel kanan (untuk HOLE ke-10 dst.)
  S               -> Skip lubang aktif (tidak disimpan)
  B               -> Kembali ke lubang sebelumnya
  N / Enter       -> Selesai untuk foto ini & lanjut
  Q / Esc         -> Keluar
"""

import cv2
import numpy as np
import csv
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── PATH ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
POV_DIR     = BASE_DIR / "dataset_final" / "pov_kendaraan"
IMAGES_DIR  = POV_DIR / "images"
LABELS_DIR  = POV_DIR / "labels"
GT_IMG_DIR  = BASE_DIR / "top_down" / "images"
POTHOLE_GT  = BASE_DIR / "pothole_gt.csv"
OUTPUT_CSV  = BASE_DIR / "dataset_final" / "validation_gt.csv"

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
WINDOW_NAME  = "Build Validation GT"
CANVAS_W     = 1500
CANVAS_H     = 900
HEADER_H     = 50
FOOTER_H     = 45
IMG_PANEL_W  = 900          # lebar panel kiri (gambar images/)
GT_PANEL_W   = CANVAS_W - IMG_PANEL_W   # panel kanan (gt thumbnails)
IMG_AREA_H   = CANVAS_H - HEADER_H - FOOTER_H

THUMB_SLOT_H = 220          # tinggi tetap per slot thumbnail (px) — atur sesuai selera

# ─── STATE ────────────────────────────────────────────────────────────────────
state = {
    "zoom":      1.0,
    "pan_x":     0,
    "pan_y":     0,
    "panning":   False,
    "pan_start": (0, 0),
    "img_w":     0,
    "img_h":     0,
}

# Area thumbnail untuk click detection: list of (canvas_y1, canvas_y2, hole_id)
_thumb_rects  = []
# Hasil klik thumbnail dari mouse callback
_click_result = {"hole_id": None}
# Offset scroll panel kanan (px, ke bawah = positif)
_scroll_offset = 0


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_pothole_gt():
    """Baca pothole_gt.csv -> list of dicts."""
    if not POTHOLE_GT.exists():
        print(f"❌  pothole_gt.csv tidak ditemukan: {POTHOLE_GT.resolve()}")
        print("   Jalankan 1_build_pothole_gt.py terlebih dahulu!")
        sys.exit(1)
    holes = []
    with open(POTHOLE_GT, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            holes.append(dict(row))
    if not holes:
        print("❌  pothole_gt.csv kosong. Jalankan 1_build_pothole_gt.py terlebih dahulu!")
        sys.exit(1)
    return holes


def load_gt_thumbnails(holes):
    """
    Siapkan thumbnail untuk setiap HOLE dari pothole_gt.csv.
    Gambar bbox lubang di atas thumbnail gt image.
    Returns: list of dicts dengan key: hole_id, img, gt_w_cm, gt_h_cm, gt_area_cm2
    """
    thumbnails = []
    for h in holes:
        gt_img_path = GT_IMG_DIR / h["gt_image"]
        img = cv2.imread(str(gt_img_path))
        if img is None:
            # Placeholder jika gambar tidak ditemukan
            img = np.full((200, 280, 3), (40, 40, 50), dtype=np.uint8)
            cv2.putText(img, "Gambar tidak", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 120), 1)
            cv2.putText(img, "ditemukan", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 120), 1)
        else:
            # Gambar bounding box lubang
            ih, iw = img.shape[:2]
            cx  = float(h["yolo_cx"]);  cy  = float(h["yolo_cy"])
            bw  = float(h["yolo_w"]);   bh  = float(h["yolo_h"])
            x1  = int((cx - bw/2) * iw);  y1 = int((cy - bh/2) * ih)
            x2  = int((cx + bw/2) * iw);  y2 = int((cy + bh/2) * ih)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 3)
            # Overlay semi-transparan pada lubang
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), -1)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        thumbnails.append({
            "hole_id":     h["pothole_id"],
            "img":         img,
            "gt_w_cm":     float(h["gt_w_cm"]),
            "gt_h_cm":     float(h["gt_h_cm"]),
            "gt_area_cm2": float(h["gt_area_cm2"]),
        })
    return thumbnails


def load_yolo_labels(label_path):
    """Baca label YOLO (dengan atau tanpa confidence) -> list [cx, cy, w, h]."""
    boxes = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append([float(p) for p in parts[1:5]])
    return boxes


def load_existing_validation():
    """
    Baca validation_gt.csv yang ada.
    Returns:
        rows            : list semua dict baris CSV
        img_existing    : dict {image_name: {pothole_idx_in_image (1-based): hole_id}}
    """
    if not OUTPUT_CSV.exists():
        return [], {}
    rows = []
    img_existing = {}   # {image_name: {1-based pothole_idx: hole_id}}
    with open(OUTPUT_CSV, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
            img = row["real_image"]
            if img not in img_existing:
                img_existing[img] = {}
            pidx = int(row["pothole_idx_in_image"])
            img_existing[img][pidx] = row["pothole_id"]
    return rows, img_existing


# ─── KOORDINAT ────────────────────────────────────────────────────────────────

def img_to_screen(x, y):
    sx = int(x * state["zoom"] + state["pan_x"])
    sy = int(y * state["zoom"] + state["pan_y"]) + HEADER_H
    return sx, sy


# ─── MOUSE CALLBACK ───────────────────────────────────────────────────────────

def mouse_callback(event, x, y, flags, param):
    global _scroll_offset

    in_right_panel = (x >= IMG_PANEL_W)

    if event == cv2.EVENT_RBUTTONDOWN:
        if not in_right_panel:
            state["panning"]   = True
            state["pan_start"] = (x, y)

    elif event == cv2.EVENT_RBUTTONUP:
        state["panning"] = False

    elif event == cv2.EVENT_MOUSEMOVE and state["panning"]:
        dx = x - state["pan_start"][0]
        dy = y - state["pan_start"][1]
        state["pan_x"] += dx
        state["pan_y"] += dy
        state["pan_start"] = (x, y)

    elif event == cv2.EVENT_MOUSEWHEEL:
        if in_right_panel:
            # Scroll panel kanan: geser daftar thumbnail
            step = 60   # px per satu tick scroll
            _scroll_offset -= step if (flags > 0) else -step
            # Batasan bawah: 0 (tidak bisa scroll ke atas dari awal)
            _scroll_offset = max(0, _scroll_offset)
        else:
            # Scroll panel kiri: zoom gambar
            factor = 1.05 if (flags > 0) else 0.95
            state["zoom"] = max(0.2, min(5.0, state["zoom"] * factor))

    elif event == cv2.EVENT_LBUTTONDOWN:
        # Klik di panel kanan → deteksi thumbnail
        if in_right_panel:
            for cy1, cy2, hole_id in _thumb_rects:
                if cy1 <= y <= cy2:
                    _click_result["hole_id"] = hole_id
                    break


# ─── RENDER PANEL KIRI ────────────────────────────────────────────────────────

def draw_left_panel(canvas, base_img, boxes, assignments, current_idx):
    """Panel kiri: gambar images/ dengan semua bbox, lubang aktif disorot kuning."""
    img_w = state["img_w"]
    img_h = state["img_h"]
    area_y1, area_y2 = HEADER_H, CANVAS_H - FOOTER_H

    # Render gambar (clip ke panel kiri)
    new_w   = max(1, int(img_w * state["zoom"]))
    new_h   = max(1, int(img_h * state["zoom"]))
    zoomed  = cv2.resize(base_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ox      = int(state["pan_x"])
    oy      = int(state["pan_y"]) + HEADER_H
    x1_c    = max(0, ox);                   y1_c = max(area_y1, oy)
    x2_c    = min(IMG_PANEL_W, ox + new_w); y2_c = min(area_y2, oy + new_h)
    x1_z    = max(0, -ox);                  y1_z = max(0, area_y1 - oy)
    x2_z    = x1_z + (x2_c - x1_c);        y2_z = y1_z + (y2_c - y1_c)
    if x2_c > x1_c and y2_c > y1_c and x2_z <= new_w and y2_z <= new_h:
        canvas[y1_c:y2_c, x1_c:x2_c] = zoomed[y1_z:y2_z, x1_z:x2_z]

    # Gambar bounding box tiap lubang
    for i, (cx, cy, bw, bh) in enumerate(boxes):
        sx1, sy1 = img_to_screen((cx - bw/2) * img_w, (cy - bh/2) * img_h)
        sx2, sy2 = img_to_screen((cx + bw/2) * img_w, (cy + bh/2) * img_h)
        # Clip x ke panel kiri
        sx1 = max(0, min(sx1, IMG_PANEL_W - 1))
        sx2 = max(0, min(sx2, IMG_PANEL_W - 1))

        if i == current_idx:
            # Aktif → kuning terang + tebal
            color, thick = (0, 215, 255), 3
            label = f"Lubang #{i+1}  <- pilih HOLE di kanan"
        elif i in assignments:
            # Sudah diassign → hijau
            color, thick = (50, 210, 80), 2
            label = f"#{i+1} -> {assignments[i]}"
        else:
            # Belum → biru redup
            color, thick = (80, 80, 180), 1
            label = f"#{i+1} (menunggu)"

        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), color, thick)
        cv2.putText(canvas, label,
                    (sx1, max(sy1 - 8, area_y1 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

    # Garis pembatas kanan
    cv2.line(canvas, (IMG_PANEL_W, area_y1), (IMG_PANEL_W, area_y2), (55, 55, 65), 2)


# ─── RENDER PANEL KANAN ───────────────────────────────────────────────────────

def draw_right_panel(canvas, thumbnails, current_hole_id=None):
    """Panel kanan: thumbnail gt images, scrollable, tinggi tetap per slot."""
    global _thumb_rects
    _thumb_rects = []

    x_off    = IMG_PANEL_W + 2
    area_y1  = HEADER_H
    area_y2  = CANVAS_H - FOOTER_H
    panel_w  = GT_PANEL_W - 6
    sub_h    = 28        # tinggi sub-header
    pad      = 6
    txt_h    = 44        # ruang teks di bawah thumbnail
    sb_w     = 8         # lebar scrollbar
    thumb_w  = panel_w - pad * 2 - sb_w - 4

    n = len(thumbnails)
    if n == 0:
        return

    # Total tinggi semua slot
    total_slots_h = n * THUMB_SLOT_H
    avail_h       = area_y2 - area_y1 - sub_h

    # Batasi scroll agar tidak melewati ujung bawah
    max_scroll = max(0, total_slots_h - avail_h)
    offset     = min(_scroll_offset, max_scroll)

    # Sub-header
    cv2.rectangle(canvas, (x_off, area_y1), (CANVAS_W, area_y1 + sub_h), (28, 28, 42), -1)
    scroll_hint = "  ^v scroll" if total_slots_h > avail_h else ""
    cv2.putText(canvas,
                f"Ground Truth Reference - klik atau tekan angka{scroll_hint}",
                (x_off + 8, area_y1 + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 190), 1)

    # Clip area gambar ke panel kanan saja
    content_y1 = area_y1 + sub_h
    content_y2 = area_y2

    for i, t in enumerate(thumbnails):
        hole_id  = t["hole_id"]
        key_num  = i + 1
        selected = (hole_id == current_hole_id)

        # Posisi slot dalam canvas (dengan offset scroll)
        y_top = content_y1 + i * THUMB_SLOT_H - offset
        y_bot = y_top + THUMB_SLOT_H - 3

        # Skip jika slot sepenuhnya di luar area tampil
        if y_bot < content_y1 or y_top > content_y2:
            continue

        # Clip y ke area konten
        draw_y_top = max(y_top, content_y1)
        draw_y_bot = min(y_bot, content_y2)

        # Background
        bg_col     = (38, 72, 38)  if selected else (22, 22, 32)
        brd_col    = (60, 220, 80) if selected else (55, 55, 70)
        brd_thick  = 2 if selected else 1
        cv2.rectangle(canvas, (x_off, draw_y_top), (CANVAS_W - sb_w - 2, draw_y_bot),
                      bg_col, -1)
        cv2.rectangle(canvas, (x_off, draw_y_top), (CANVAS_W - sb_w - 2, draw_y_bot),
                      brd_col, brd_thick)

        # ── Thumbnail image ──
        img_area_h = THUMB_SLOT_H - pad * 2 - txt_h
        if img_area_h > 10 and thumb_w > 10:
            src    = t["img"]
            ih, iw = src.shape[:2]
            scale  = min(thumb_w / iw, img_area_h / ih)
            tw     = max(1, int(iw * scale))
            th     = max(1, int(ih * scale))
            thumb  = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA)

            ty1 = y_top + pad
            tx1 = x_off + pad + (thumb_w - tw) // 2
            ty2 = ty1 + th
            tx2 = tx1 + tw

            # Clip thumbnail ke area konten sebelum ditempel
            src_y0 = max(0, content_y1 - ty1)
            src_y1 = th - max(0, ty2 - content_y2)
            dst_y0 = max(ty1, content_y1)
            dst_y1 = min(ty2, content_y2)
            if dst_y1 > dst_y0 and src_y1 > src_y0 and tx2 <= CANVAS_W and tx1 >= 0:
                canvas[dst_y0:dst_y1, tx1:tx2] = thumb[src_y0:src_y1, :]

        # ── Teks info (hanya jika cukup ruang) ──
        txt_y = y_bot - txt_h + 14
        if content_y1 < txt_y < content_y2:
            key_col = (60, 230, 80) if selected else (210, 210, 210)
            cv2.putText(canvas,
                        f"[{key_num}]  {hole_id}",
                        (x_off + 8, txt_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, key_col, 2)
        if content_y1 < txt_y + 20 < content_y2:
            cv2.putText(canvas,
                        f"{t['gt_w_cm']:.1f} x {t['gt_h_cm']:.1f} cm  ({t['gt_area_cm2']:.1f} cm2)",
                        (x_off + 8, txt_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 175, 140), 1)

        # Simpan area click detection (koordinat canvas absolut, clipped)
        _thumb_rects.append((draw_y_top, draw_y_bot, hole_id))

    # ── Scrollbar ──
    if total_slots_h > avail_h:
        sb_x1 = CANVAS_W - sb_w
        sb_x2 = CANVAS_W - 1
        # Track background
        cv2.rectangle(canvas, (sb_x1, content_y1), (sb_x2, content_y2), (35, 35, 45), -1)
        # Thumb proporsi
        thumb_ratio = avail_h / total_slots_h
        thumb_h_sb  = max(20, int(avail_h * thumb_ratio))
        thumb_pos   = int((offset / max_scroll) * (avail_h - thumb_h_sb)) if max_scroll > 0 else 0
        sb_ty1 = content_y1 + thumb_pos
        sb_ty2 = sb_ty1 + thumb_h_sb
        cv2.rectangle(canvas, (sb_x1 + 1, sb_ty1), (sb_x2 - 1, sb_ty2), (100, 130, 100), -1)


# ─── RENDER CANVAS UTAMA ──────────────────────────────────────────────────────

def draw_canvas(base_img, boxes, assignments, current_idx,
                thumbnails, img_name, img_idx, total_imgs):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # ── Header ──
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, HEADER_H), (18, 18, 30), -1)
    progress  = f"[{img_idx+1}/{total_imgs}]"
    lubang_st = f"Lubang {current_idx+1}/{len(boxes)}"
    assigned  = len([k for k in assignments if assignments[k] is not None])
    cv2.putText(canvas,
                f"{progress}  {img_name[:65]}     {lubang_st}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
    cv2.putText(canvas,
                f"Sudah di-assign: {assigned}/{len(boxes)}  |  Zoom: {state['zoom']:.1f}x  |  "
                f"Klik-kanan drag: geser",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 200, 140), 1)

    # ── Panels ──
    draw_left_panel(canvas, base_img, boxes, assignments, current_idx)
    draw_right_panel(canvas, thumbnails, assignments.get(current_idx))

    # ── Footer ──
    cv2.rectangle(canvas, (0, CANVAS_H - FOOTER_H), (CANVAS_W, CANVAS_H), (18, 18, 30), -1)
    cur_assign = assignments.get(current_idx)
    if cur_assign:
        foot_txt = (f"Lubang #{current_idx+1} -> {cur_assign}  "
                    f"(ubah: tekan angka lain  |  lanjut lubang berikutnya: tekan angka HOLE-nya  |  "
                    f"simpan & lanjut gambar: N / Enter)")
        foot_col = (55, 230, 80)
    else:
        n = len(thumbnails)
        foot_txt = (f"Lubang #{current_idx+1}: Tekan 1-{n} atau klik thumbnail  |  "
                    f"S: skip  |  B: kembali  |  N/Enter: simpan gambar ini  |  Q: keluar")
        foot_col = (50, 170, 255)
    cv2.putText(canvas, foot_txt, (8, CANVAS_H - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, foot_col, 1)
    cv2.putText(canvas,
                f"Scroll kanan: geser daftar  |  Scroll kiri: zoom  |  Klik-kanan drag: geser gambar  |  Q: keluar",
                (8, CANVAS_H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 110, 110), 1)

    return canvas


# ─── SIMPAN CSV ───────────────────────────────────────────────────────────────

FIELDNAMES = [
    "real_id", "real_image", "pothole_idx_in_image",
    "pothole_id", "gt_w_cm", "gt_h_cm", "gt_area_cm2",
]


def real_num(real_id):
    """REAL_042 -> 42"""
    try:
        return int(str(real_id).split("_")[-1])
    except ValueError:
        return 0


def ask(prompt, default=""):
    try:
        ans = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    return ans or default


def write_validation_csv(rows):
    rows = sorted(rows, key=lambda r: real_num(r["real_id"]))
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅  Disimpan ke: {OUTPUT_CSV.resolve()}  ({len(rows)} baris)")


def sync_sizes(existing_rows, hole_lookup):
    """Mode [S]: salin ukuran terbaru dari pothole_gt.csv tanpa membuka jendela."""
    changes = []
    for r in existing_rows:
        h = hole_lookup.get(r["pothole_id"])
        if h is None:
            print(f"   ⚠  {r['real_id']}: {r['pothole_id']} tidak ada di pothole_gt.csv")
            continue
        old = (r["gt_w_cm"], r["gt_h_cm"], r["gt_area_cm2"])
        new = (h["gt_w_cm"], h["gt_h_cm"], h["gt_area_cm2"])
        if tuple(map(str, old)) != tuple(map(str, new)):
            changes.append((r["real_id"], r["pothole_id"], old, new))
            r["gt_w_cm"], r["gt_h_cm"], r["gt_area_cm2"] = new
    if not changes:
        print("✅  Semua ukuran sudah sama dengan pothole_gt.csv. Tidak ada perubahan.")
        return
    print(f"\n   {len(changes)} baris akan berubah:")
    for rid, hid, old, new in changes[:20]:
        print(f"   {rid} ({hid}): {old[0]} x {old[1]}  ->  {new[0]} x {new[1]} cm")
    if len(changes) > 20:
        print(f"   ... dan {len(changes) - 20} baris lain")
    if ask("   Simpan ke validation_gt.csv? (y/N): ", "n").lower() == "y":
        write_validation_csv(existing_rows)
    else:
        print("   Tidak disimpan.")


def choose_images(image_files, img_existing_asgn):
    listed = [f for f in image_files if f.name in img_existing_asgn]
    print("\n   Foto yang sudah tercatat:")
    for i, f in enumerate(listed, 1):
        asg = img_existing_asgn[f.name]
        holes = ", ".join(asg[k] for k in sorted(asg))
        print(f"   [{i:2d}] {f.name[:40]}  ->  {holes}")
    ans = ask("   Nomor foto (contoh: 3 atau 1,4,7): ")
    chosen = []
    for part in ans.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= len(listed):
            chosen.append(listed[int(part) - 1])
    return chosen


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    global _scroll_offset

    if not IMAGES_DIR.exists():
        print(f"❌  Folder tidak ditemukan: {IMAGES_DIR.resolve()}")
        sys.exit(1)

    # ── Load data referensi ──
    holes       = load_pothole_gt()
    hole_lookup = {h["pothole_id"]: h for h in holes}

    # ── Daftar gambar ──
    image_files = sorted(
        list(IMAGES_DIR.glob("*.jpg"))  +
        list(IMAGES_DIR.glob("*.jpeg")) +
        list(IMAGES_DIR.glob("*.png"))
    )
    if not image_files:
        print(f"❌  Tidak ada gambar di: {IMAGES_DIR.resolve()}")
        sys.exit(1)

    existing_rows, img_existing_asgn = load_existing_validation()
    next_real = max([real_num(r["real_id"]) for r in existing_rows], default=0)

    print("=" * 65)
    print("  LANGKAH 3  -  BUILD VALIDATION GT")
    print("=" * 65)
    print(f"📂  Foto POV kendaraan : {IMAGES_DIR.resolve()}")
    print(f"📐  Ukuran lubang      : {POTHOLE_GT.resolve()}")
    print(f"💾  Output             : {OUTPUT_CSV.resolve()}")
    print(f"📐  Daftar HOLE:")
    for i, h in enumerate(holes):
        print(f"   [{i+1}] {h['pothole_id']}: {h['gt_w_cm']} x {h['gt_h_cm']} cm")

    mode = "L"
    if existing_rows:
        print(f"\n📋  validation_gt.csv sudah berisi {len(existing_rows)} anotasi "
              f"dari {len(img_existing_asgn)} foto.")
        print("   [L] Lanjut   - hanya anotasi yang belum dipetakan (default)")
        print("   [U] Ubah     - pilih foto tertentu dan ubah pemetaan HOLE-nya")
        print("   [S] Sinkron  - perbarui ukuran dari pothole_gt.csv (tanpa jendela)")
        mode = ask("   Pilihan (L/U/S): ", "L").upper()

    if mode == "S":
        sync_sizes(existing_rows, hole_lookup)
        return

    if mode == "U":
        remaining = choose_images(image_files, img_existing_asgn)
    else:
        remaining = []
        for f in image_files:
            n_lbl  = len(load_yolo_labels(LABELS_DIR / (f.stem + ".txt")))
            n_done = len(img_existing_asgn.get(f.name, {}))
            if n_lbl > n_done:
                remaining.append(f)

    print(f"🖼  Akan diproses: {len(remaining)} foto")
    print("─" * 65)
    if not remaining:
        print("✅  Tidak ada foto yang perlu diproses.")
        return

    thumbnails = load_gt_thumbnails(holes)
    results    = {}    # {image_name: {pidx 0-based: hole_id}}

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    for img_idx, img_path in enumerate(remaining):
        state.update({"zoom": 1.0, "pan_x": 0, "pan_y": 0, "panning": False})
        _scroll_offset = 0

        base_img = cv2.imread(str(img_path))
        if base_img is None:
            print(f"⚠  Gagal membuka: {img_path.name}")
            continue

        state["img_h"], state["img_w"] = base_img.shape[:2]
        img_w, img_h = state["img_w"], state["img_h"]

        fit_zoom = min(IMG_PANEL_W / img_w, IMG_AREA_H / img_h)
        state["zoom"]  = fit_zoom
        state["pan_x"] = int((IMG_PANEL_W - img_w * fit_zoom) / 2)
        state["pan_y"] = int((IMG_AREA_H  - img_h * fit_zoom) / 2)

        boxes = load_yolo_labels(LABELS_DIR / (img_path.stem + ".txt"))
        if not boxes:
            print(f"⚠  [{img_idx+1}/{len(remaining)}] Tidak ada label: {img_path.name} - skip")
            continue

        print(f"\n🖼  [{img_idx+1}/{len(remaining)}] {img_path.name}  |  {len(boxes)} lubang")

        prior = img_existing_asgn.get(img_path.name, {})
        assignments = {(pidx - 1): hole_id for pidx, hole_id in prior.items()}
        skipped_set = set()
        if mode == "U":
            current_idx = 0
            print("   ↩  Pemetaan lama dimuat. Tekan angka / klik thumbnail untuk mengganti.")
        else:
            current_idx = next((i for i in range(len(boxes)) if i not in assignments), 0)
            if prior:
                print(f"   ↩  {len(prior)} lubang sudah ada, mulai dari lubang #{current_idx+1}")
        action = "continue"

        while True:
            _click_result["hole_id"] = None

            canvas = draw_canvas(
                base_img, boxes, assignments, current_idx,
                thumbnails, img_path.name, img_idx, len(remaining)
            )
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKeyEx(30)

            if _click_result["hole_id"] is not None:
                selected_hole = _click_result["hole_id"]
                _click_result["hole_id"] = None
                assignments[current_idx] = selected_hole
                print(f"   🖱  Lubang #{current_idx+1} -> {selected_hole}")
                if current_idx < len(boxes) - 1:
                    current_idx += 1
                continue

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                action = "quit"
                break

            if ord('1') <= key <= ord('9'):
                hole_n = key - ord('0')
                if 1 <= hole_n <= len(thumbnails):
                    selected_hole = thumbnails[hole_n - 1]["hole_id"]
                    assignments[current_idx] = selected_hole
                    print(f"   ✅  Lubang #{current_idx+1} -> {selected_hole}")
                    if current_idx < len(boxes) - 1:
                        current_idx += 1
                else:
                    print(f"   ⚠  Tidak ada HOLE #{hole_n} (total: {len(thumbnails)})")

            elif key in (ord('s'), ord('S')):
                skipped_set.add(current_idx)
                assignments.pop(current_idx, None)
                print(f"   ⏭  Lubang #{current_idx+1} di-skip")
                if current_idx < len(boxes) - 1:
                    current_idx += 1

            elif key in (ord('b'), ord('B')):
                if current_idx > 0:
                    current_idx -= 1
                    print(f"   ↩  Kembali ke lubang #{current_idx+1}")
                else:
                    print("   ⚠  Sudah di lubang pertama")

            elif key in (ord('n'), ord('N'), 13):
                if not assignments and len(skipped_set) < len(boxes):
                    print("   ⚠  Belum ada lubang yang dipetakan atau di-skip.")
                else:
                    action = "next"
                    break

            elif key in (ord('q'), ord('Q'), 27):
                action = "quit"
                break

        if action == "next" or (action == "quit" and assignments):
            results[img_path.name] = dict(assignments)
            for pidx in sorted(assignments):
                old = prior.get(pidx + 1)
                note = "" if old is None or old == assignments[pidx] else f"  (sebelumnya {old})"
                print(f"   💾  lubang #{pidx+1} -> {assignments[pidx]}{note}")

        if action == "quit":
            print("\n🚪  Keluar...")
            break

    cv2.destroyAllWindows()

    if not results:
        print("\n⚠  Tidak ada data baru. validation_gt.csv tidak diubah.")
        return

    # ── Gabungkan ke baris lama. REAL_ID lama dipertahankan untuk anotasi yang sama ──
    old_by_key = {(r["real_image"], int(r["pothole_idx_in_image"])): r for r in existing_rows}
    combined   = [r for r in existing_rows if r["real_image"] not in results]
    for img_name, asg in results.items():
        for pidx in sorted(asg):
            hole_id = asg[pidx]
            h   = hole_lookup[hole_id]
            old = old_by_key.get((img_name, pidx + 1))
            if old is not None:
                real_id = old["real_id"]
            else:
                next_real += 1
                real_id = f"REAL_{next_real:03d}"
            combined.append({
                "real_id":              real_id,
                "real_image":           img_name,
                "pothole_idx_in_image": pidx + 1,
                "pothole_id":           hole_id,
                "gt_w_cm":              h["gt_w_cm"],
                "gt_h_cm":              h["gt_h_cm"],
                "gt_area_cm2":          h["gt_area_cm2"],
            })

    print("\n" + "─" * 65)
    print(f"   {len(results)} foto diproses. Total baris setelah disimpan: {len(combined)}")
    if ask("   Simpan ke validation_gt.csv? (y/N): ", "n").lower() == "y":
        write_validation_csv(combined)
    else:
        print("   Tidak disimpan. validation_gt.csv tetap seperti semula.")
    print("\nSelesai!")


if __name__ == "__main__":
    main()
