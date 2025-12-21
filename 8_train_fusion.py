# 8_train_fusion.py
"""
Safe/resumable Fusion trainer.
Inputs:
 - FEATURES_CSV (features_windows.csv) : windows with features
 - LABELS_CSV   (fusion_labels.csv)    : window_start_s, window_end_s, drowsy (0/1)
Outputs:
 - WEIGHTS_OUT (weights/fusion_model.h5)
 - SCALER_OUT  (weights/fusion_scaler.pkl)
 - HISTORY_OUT (weights/fusion_history.json)
 - REPORT_OUT  (weights/fusion_report.txt)
 - PRED_OUT    (weights/fusion_val_predictions.csv)  (optional)
"""
import json, pickle, os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# CONFIG
FEATURES_CSV = "features_windows.csv"
LABELS_CSV   = "fusion_labels.csv"
WEIGHTS_OUT  = Path("weights") / "fusion_model.h5"
SCALER_OUT   = Path("weights") / "fusion_scaler.pkl"
HISTORY_OUT  = Path("weights") / "fusion_history.json"
REPORT_OUT   = Path("weights") / "fusion_report.txt"
PRED_OUT     = Path("weights") / "fusion_val_predictions.csv"
Path("weights").mkdir(exist_ok=True)

EPOCHS = 200
BATCH = 32
RANDOM_STATE = 42
TEST_SIZE = 0.2

def build_model(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(32, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def atomic_write_json(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))

def train():
    # 1. checks
    if not Path(FEATURES_CSV).exists():
        raise FileNotFoundError(f"Missing features CSV: {FEATURES_CSV}")
    if not Path(LABELS_CSV).exists():
        raise FileNotFoundError(f"Missing labels CSV: {LABELS_CSV}")

    feats = pd.read_csv(FEATURES_CSV)
    labs  = pd.read_csv(LABELS_CSV)

    # ensure required columns exist
    req_fcols = {"window_start_s","window_end_s","perclos","max_closed_dur_s","yawn_count_60s","avg_yawn_prob","avg_head_tilt","avg_shoulder"}
    if not req_fcols.issubset(set(feats.columns)):
        raise ValueError(f"Features CSV missing columns. Need: {sorted(req_fcols)}")
    if not {"window_start_s","window_end_s","drowsy"}.issubset(set(labs.columns)):
        raise ValueError("Labels CSV must contain: window_start_s, window_end_s, drowsy")

    # 2. merge
    merged = pd.merge(feats, labs, on=["window_start_s","window_end_s"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping windows between features and labels. Check alignment of window_start_s/window_end_s.")

    X = merged[["perclos","max_closed_dur_s","yawn_count_60s","avg_yawn_prob","avg_head_tilt","avg_shoulder"]].values
    y = merged["drowsy"].astype(int).values

    # 3. scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 4. split (stratify if possible)
    stratify = y if len(np.unique(y)) > 1 else None
    Xtr, Xv, ytr, yv = train_test_split(Xs, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify)

    # 5. build model
    model = build_model(X.shape[1])
    ck = ModelCheckpoint(str(WEIGHTS_OUT), save_best_only=True, monitor='val_accuracy', mode='max')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 6. train
    history = model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=EPOCHS, batch_size=BATCH, callbacks=[ck, es], verbose=2)

    # 7. eval
    preds = (model.predict(Xv).reshape(-1) > 0.5).astype(int)
    report = classification_report(yv, preds, digits=4)
    cm = confusion_matrix(yv, preds)

    # 8. save everything
    # model already saved by ModelCheckpoint (best). Also save final model explicitly.
    model.save(str(WEIGHTS_OUT))

    # scaler
    with open(str(SCALER_OUT), "wb") as f:
        pickle.dump(scaler, f)

    # history -> make JSON serializable
    hist_obj = {k: [float(x) for x in v] for k,v in history.history.items()}
    atomic_write_json(hist_obj, HISTORY_OUT)

    # report + confusion + metadata
    with open(str(REPORT_OUT), "w") as f:
        f.write("Fusion model training report\n\n")
        f.write("=== Classification report (val) ===\n")
        f.write(report + "\n\n")
        f.write("=== Confusion matrix ===\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write("=== Data info ===\n")
        f.write(f"Total windows (features): {len(feats)}\n")
        f.write(f"Total merged windows used: {len(merged)}\n")
        f.write(f"Train shape: {Xtr.shape}, Val shape: {Xv.shape}\n")
        f.write("History keys: " + json.dumps(list(history.history.keys())) + "\n")
    # predictions CSV
    df_val = pd.DataFrame(Xv, columns=["perclos","max_closed_dur_s","yawn_count_60s","avg_yawn_prob","avg_head_tilt","avg_shoulder"])
    df_val["y_true"] = yv
    df_val["y_pred"] = preds
    df_val.to_csv(str(PRED_OUT), index=False)

    print("[done] weights:", WEIGHTS_OUT)
    print("[done] scaler:", SCALER_OUT)
    print("[done] history:", HISTORY_OUT)
    print("[done] report:", REPORT_OUT)
    print("[done] val predictions:", PRED_OUT)

    return {
        "weights": str(WEIGHTS_OUT),
        "scaler": str(SCALER_OUT),
        "history": str(HISTORY_OUT),
        "report": str(REPORT_OUT),
        "predictions": str(PRED_OUT)
    }

if __name__ == "__main__":
    train()
