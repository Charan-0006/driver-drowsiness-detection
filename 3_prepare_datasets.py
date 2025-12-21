# 3_prepare_datasets.py
"""
Prepare per-task datasets (eye, mouth(yawn), head) from per_frame_annotations.csv.
Crops are center-based. Uses deterministic subject-based split (~20% val).
Outputs:
  datasets_cnn/
    eye/
      train/open
      train/closed
      val/open
      val/closed
    mouth/
      train/no_yawn
      train/yawn
      val/...
    head/
      train/normal
      train/tilt
      val/...
This version prints progress every PRINT_EVERY frames and skips already saved files.
"""
import pandas as pd
import cv2
from pathlib import Path
from collections import Counter


ANNOT_CSV = "per_frame_annotations.csv"
FRAMES_ROOT = r"E:\YawDD\YawDD_frames"   # where frames are stored (not strictly used here)
OUT_DATA_ROOT = "datasets_cnn"
IMG_SIZE = (224, 224)
PRINT_EVERY = 500          # change to smaller value for more frequent prints
# -------------------------------

TASKS = {
    "eye": ("open", "closed"),
    "mouth": ("no_yawn", "yawn"),   # mouth corresponds to 'yawn' column in CSV
    "head": ("normal", "tilt")
}

def subject_to_split(subject_name):
    if not subject_name:
        return "train"
    ssum = sum(ord(c) for c in subject_name)
    return "val" if (ssum % 10) < 2 else "train"

def ensure_dirs(root):
    root = Path(root)
    for task, classes in TASKS.items():
        for split in ("train","val"):
            for c in classes:
                (root / task / split / c).mkdir(parents=True, exist_ok=True)

def crop_center_face(img):
    h,w = img.shape[:2]
    fw = int(w*0.6); fh = int(h*0.6)
    cx,cy = w//2, h//2
    x1 = max(0, cx - fw//2); y1 = max(0, cy - fh//2)
    x2 = min(w, x1 + fw); y2 = min(h, y1 + fh)
    return img[y1:y2, x1:x2]

def prepare_datasets(annot_csv=ANNOT_CSV, outroot=OUT_DATA_ROOT):
    df = pd.read_csv(annot_csv)
    outroot = Path(outroot)
    ensure_dirs(outroot)

    total_processed = 0
    total_saved = 0
    skipped = 0
    errors = 0
    counts = Counter()

    for idx, row in df.iterrows():
        total_processed += 1
        # frame path in CSV may be absolute; if not, you may need to join with FRAMES_ROOT
        frame_path = Path(row["frame"])
        if not frame_path.exists():
            # try joining with FRAMES_ROOT
            candidate = Path(FRAMES_ROOT) / frame_path.name
            if candidate.exists():
                frame_path = candidate
            else:
                # skip missing file
                errors += 1
                if total_processed % PRINT_EVERY == 0:
                    print(f"[{total_processed}] missing frame -> {row['frame']}")
                continue

        # infer subject / split
        # prefer using video-folder name as subject: frames_root/<camera>/<gender>/<video_folder>/<frame.jpg>
        try:
            rel = frame_path.relative_to(Path(FRAMES_ROOT))
            rel_parts = rel.parts
        except Exception:
            rel_parts = frame_path.parts
        # choose video folder name if available
        subject = rel_parts[-2] if len(rel_parts) >= 2 else frame_path.parent.name
        split = subject_to_split(subject)

        # read image
        img = cv2.imread(str(frame_path))
        if img is None:
            errors += 1
            if total_processed % PRINT_EVERY == 0:
                print(f"[{total_processed}] failed to load image {frame_path}")
            continue

        # crops
        try:
            face = crop_center_face(img)
            hf = face.shape[0] if face is not None else 0
            if hf <= 8:
                # tiny/invalid crop
                errors += 1
                continue
            eye_crop = face[0: max(1, hf//3), :]
            mouth_crop = face[max(1, hf//2): min(hf, hf*3//4), :]
            head_crop = face.copy()

            eye_r = cv2.resize(eye_crop, IMG_SIZE)
            mouth_r = cv2.resize(mouth_crop, IMG_SIZE)
            head_r = cv2.resize(head_crop, IMG_SIZE)
        except Exception:
            errors += 1
            continue

        base = frame_path.stem

        # labels (ensure these column names exist in your CSV)
        eye_label = "closed" if int(row.get("eye_closed", 0)) == 1 else "open"
        mouth_label = "yawn" if int(row.get("yawn", 0)) == 1 else "no_yawn"
        head_label = "tilt" if int(row.get("head_tilt", 0)) == 1 else "normal"

        # prepare destination paths
        p_eye = outroot / "eye" / split / eye_label / f"{base}.jpg"
        p_mouth = outroot / "mouth" / split / mouth_label / f"{base}.jpg"
        p_head = outroot / "head" / split / head_label / f"{base}.jpg"

        # skip if already exists
        if p_eye.exists() and p_mouth.exists() and p_head.exists():
            skipped += 1
        else:
            try:
                cv2.imwrite(str(p_eye), eye_r)
                cv2.imwrite(str(p_mouth), mouth_r)
                cv2.imwrite(str(p_head), head_r)
                total_saved += 1
                counts[f"eye/{split}/{eye_label}"] += 1
                counts[f"mouth/{split}/{mouth_label}"] += 1
                counts[f"head/{split}/{head_label}"] += 1
            except Exception as e:
                errors += 1

        # periodic print
        if total_processed % PRINT_EVERY == 0:
            print(f"[{total_processed}] processed, saved so far: {total_saved}, skipped: {skipped}, errors: {errors}")

    # final summary
    print("=== DONE ===")
    print("Total processed rows:", total_processed)
    print("Total saved images (approx):", total_saved)
    print("Total skipped (already existed):", skipped)
    print("Total errors:", errors)
    print("Per-class counts (sample):")
    for k,v in counts.most_common(30):
        print(f"  {k}: {v}")
    print("Datasets created under:", outroot)

if __name__ == "__main__":
    prepare_datasets()
