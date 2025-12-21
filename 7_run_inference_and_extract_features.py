# 7_infer_and_make_features_resume_gpu.py
"""
GPU-optimized resume-capable inference with mixed precision and threaded image loading.

EDIT SETTINGS in the CONFIG block below.
"""
import os, re, time, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# -------------------- CONFIG --------------------
FRAMES_ROOT = Path(r"E:\YawDD\YawDD_frames")
WEIGHTS_DIR = Path("weights")
EYE_MODEL = WEIGHTS_DIR / "eye_model.h5"
YAWN_MODEL = WEIGHTS_DIR / "mouth_model.h5"
HEAD_MODEL = WEIGHTS_DIR / "head_model.h5"
SHOULDER_MODEL = WEIGHTS_DIR / "shoulder_model.h5"  # optional
IMG_SIZE = (224, 224)

PER_FRAME_CSV = Path("per_frame_predictions.csv")
FEATURES_CSV = Path("features_windows.csv")

WINDOW_S = 10
STRIDE_S = 5

# Performance tuning
BATCH_SIZE = 128            # increase with GPU memory (128/256/512)
SAVE_EVERY = 2000           # flush after this many new rows
NUM_WORKERS = min(8, os.cpu_count() or 4)
USE_MIXED_PRECISION = True  # set False if you want standard FP32
FORCE_RESTART = False
# -------------------------------------------------

# ------------- TF GPU & mixed precision setup -------------
physical_gpus = tf.config.list_physical_devices("GPU")
if physical_gpus:
    try:
        for g in physical_gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("[INFO] Mixed precision enabled (policy=mixed_float16)")
else:
    print("[INFO] Mixed precision disabled (using float32)")

print("[INFO] GPUs:", physical_gpus)

# ---------------- Load models safely ----------------
def load_model_safe(path: Path):
    if path is None or not path.exists():
        print(f"[INFO] model not found: {path}")
        return None
    try:
        m = tf.keras.models.load_model(str(path))
        print(f"[OK] loaded model: {path}")
        return m
    except Exception as e:
        print(f"[WARN] failed to load {path}: {e}")
        return None

models = {
    "eye": load_model_safe(EYE_MODEL),
    "yawn": load_model_safe(YAWN_MODEL),
    "head": load_model_safe(HEAD_MODEL),
    "shoulder": load_model_safe(SHOULDER_MODEL)
}

# ---------------- Utilities ----------------
def write_csv_atomic(df: pd.DataFrame, path: Path):
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False)
    shutil.move(tmp, str(path))

def load_existing(per_frame_csv: Path):
    if per_frame_csv.exists() and not FORCE_RESTART:
        try:
            df = pd.read_csv(per_frame_csv)
            processed = set(df['frame'].astype(str).tolist()) if 'frame' in df.columns else set()
            print(f"[INFO] loaded existing CSV ({len(df)} rows) -> skipping {len(processed)} frames")
            return processed, df
        except Exception as e:
            print("[WARN] failed to read existing CSV:", e)
            return set(), None
    else:
        if per_frame_csv.exists() and FORCE_RESTART:
            per_frame_csv.unlink()
            print("[INFO] FORCE_RESTART: removed existing per-frame CSV")
        return set(), None

def get_all_frames(root: Path):
    files = sorted(root.rglob("*.jpg"))
    print(f"[INFO] found {len(files)} jpg files under {root}")
    return files

# threaded image loader + preproc
def load_and_preprocess(path: Path):
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        return img
    except Exception:
        return None

def parse_ts(path: Path):
    s = path.stem
    m = re.search(r"_t(\d+)ms", s)
    if m:
        try:
            return float(int(m.group(1))) / 1000.0
        except:
            pass
    try:
        return float(path.stat().st_mtime)
    except:
        return None

# Batch predict helper: accepts list[Path]
def batch_predict(models, paths_batch):
    # load & preprocess with thread pool
    imgs = [None] * len(paths_batch)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        future_to_idx = {ex.submit(load_and_preprocess, p): i for i,p in enumerate(paths_batch)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                imgs[i] = fut.result()
            except Exception:
                imgs[i] = None

    # prepare numpy array for valid images
    valid_idx = [i for i,img in enumerate(imgs) if img is not None]
    if len(valid_idx) == 0:
        # nothing could be read in this batch
        return [{
            "frame": str(paths_batch[i]),
            "ts_s": parse_ts(paths_batch[i]),
            "prob_eye": 0.0, "prob_yawn": 0.0, "prob_head": 0.0, "prob_shoulder": 0.0
        } for i in range(len(paths_batch))]

    arr = np.stack([imgs[i] for i in valid_idx], axis=0)

    def safe_batch_pred(model, arr_input):
        if model is None:
            return np.zeros((arr_input.shape[0],), dtype=np.float32)
        try:
            preds = model.predict(arr_input, batch_size=min(256, arr_input.shape[0]), verbose=0)
            preds = np.asarray(preds).ravel()
            # if mixed precision, preds may be float16; convert & sanitize
            preds = preds.astype("float32")
            preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
            return preds
        except Exception as e:
            print("[WARN] model.predict failed:", e)
            return np.zeros((arr_input.shape[0],), dtype=np.float32)

    preds_eye = safe_batch_pred(models.get("eye"), arr)
    preds_yawn = safe_batch_pred(models.get("yawn"), arr)
    preds_head = safe_batch_pred(models.get("head"), arr)
    preds_sh = safe_batch_pred(models.get("shoulder"), arr)

    # scatter results to full batch length
    out = []
    for i in range(len(paths_batch)):
        if i in valid_idx:
            j = valid_idx.index(i)
            pe = float(preds_eye[j]) if j < len(preds_eye) else 0.0
            py = float(preds_yawn[j]) if j < len(preds_yawn) else 0.0
            ph = float(preds_head[j]) if j < len(preds_head) else 0.0
            ps = float(preds_sh[j]) if j < len(preds_sh) else 0.0
        else:
            pe = py = ph = ps = 0.0
        out.append({
            "frame": str(paths_batch[i]),
            "ts_s": parse_ts(paths_batch[i]),
            "prob_eye": pe, "prob_yawn": py, "prob_head": ph, "prob_shoulder": ps
        })
    return out

# Append rows in atomic fashion
def append_rows(rows, out_csv: Path):
    df_new = pd.DataFrame(rows, columns=["frame","ts_s","prob_eye","prob_yawn","prob_head","prob_shoulder"])
    if out_csv.exists():
        try:
            df_prev = pd.read_csv(out_csv)
            df_comb = pd.concat([df_prev, df_new], ignore_index=True)
        except Exception as e:
            print("[WARN] failed reading existing csv during append:", e)
            df_comb = df_new
    else:
        df_comb = df_new
    df_comb = df_comb.drop_duplicates(subset=["frame"], keep="first").sort_values("ts_s").reset_index(drop=True)
    write_csv_atomic(df_comb, out_csv)

# Main prediction loop (batching + resume)
def predict_and_append_all(models, frames_root: Path, out_csv: Path):
    all_files = get_all_frames(frames_root)
    processed_set, _ = load_existing(out_csv)
    to_process = [f for f in all_files if str(f) not in processed_set]
    N = len(to_process)
    if N == 0:
        print("[INFO] Nothing to process.")
        return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["frame","ts_s","prob_eye","prob_yawn","prob_head","prob_shoulder"])
    print(f"[INFO] processing {N} new frames in batches of {BATCH_SIZE}")
    buffer_rows = []
    processed = 0
    for i in tqdm(range(0, N, BATCH_SIZE), desc="Batch predict"):
        batch_paths = to_process[i: i + BATCH_SIZE]
        rows = batch_predict(models, batch_paths)
        buffer_rows.extend(rows)
        processed += len(rows)
        if len(buffer_rows) >= SAVE_EVERY:
            append_rows(buffer_rows, out_csv)
            print(f"[INFO] flushed {len(buffer_rows)} rows -> {out_csv}")
            buffer_rows = []
    if buffer_rows:
        append_rows(buffer_rows, out_csv)
        print(f"[INFO] final flush {len(buffer_rows)} rows -> {out_csv}")
    print("[INFO] All predictions done.")
    return pd.read_csv(out_csv)

# Window aggregation (same as before)
def window_aggregate(df, window_s=WINDOW_S, stride_s=STRIDE_S, features_csv=FEATURES_CSV):
    if df is None or df.empty:
        cols = ["window_start_s","window_end_s","perclos","max_closed_dur_s","yawn_count_60s","avg_yawn_prob","avg_head_tilt","avg_shoulder"]
        pd.DataFrame(columns=cols).to_csv(features_csv, index=False)
        print("[INFO] wrote empty features CSV")
        return pd.DataFrame(columns=cols)

    df2 = df.dropna(subset=["ts_s"]).copy()
    df2["ts_s"] = pd.to_numeric(df2["ts_s"], errors="coerce")
    df2 = df2.dropna(subset=["ts_s"]).sort_values("ts_s").reset_index(drop=True)
    if df2.empty:
        pd.DataFrame(columns=["window_start_s","window_end_s","perclos","max_closed_dur_s","yawn_count_60s","avg_yawn_prob","avg_head_tilt","avg_shoulder"]).to_csv(features_csv, index=False)
        print("[INFO] no valid timestamps")
        return pd.DataFrame()

    start = float(df2.ts_s.min()); end = float(df2.ts_s.max())
    rows=[]
    t = start
    while t + window_s <= end + 1e-6:
        wstart = t; wend = t + window_s
        sub = df2[(df2.ts_s >= wstart) & (df2.ts_s < wend)]
        if len(sub) == 0:
            t += stride_s; continue
        perclos = float((sub.prob_eye > 0.5).mean())
        max_closed = 0.0; seg_start=None; prev=None
        for _, r in sub.iterrows():
            if r.prob_eye > 0.5:
                if seg_start is None: seg_start = r.ts_s
                prev = r.ts_s
            else:
                if seg_start is not None and prev is not None:
                    d = prev - seg_start
                    if d>max_closed: max_closed = d
                    seg_start=None; prev=None
        if seg_start is not None and prev is not None:
            d = prev - seg_start
            if d>max_closed: max_closed = d
        yawns60 = df2[(df2.ts_s >= (wend-60.0)) & (df2.ts_s < wend)]
        yawn_count = int((yawns60.prob_yawn > 0.5).sum())
        avg_yawn = float(sub.prob_yawn.mean())
        avg_head = float(sub.prob_head.mean())
        avg_sh = float(sub.prob_shoulder.mean())
        rows.append({"window_start_s":wstart,"window_end_s":wend,"perclos":perclos,"max_closed_dur_s":max_closed,
                     "yawn_count_60s":yawn_count,"avg_yawn_prob":avg_yawn,"avg_head_tilt":avg_head,"avg_shoulder":avg_sh})
        t += stride_s

    out = pd.DataFrame(rows)
    write_csv_atomic(out, features_csv)
    print(f"[INFO] Saved {features_csv} ({len(out)} windows)")
    return out

# ---------------- main ----------------
def main():
    t0 = time.time()
    print("[START] GPU resume-capable inference")
    df_all = predict_and_append_all(models, FRAMES_ROOT, PER_FRAME_CSV)
    print(f"[INFO] Predictions done ({len(df_all)} rows). Time: {time.time()-t0:.1f}s")
    feats = window_aggregate(df_all, WINDOW_S, STRIDE_S, FEATURES_CSV)
    print("[END]")

if __name__ == "__main__":
    main()
