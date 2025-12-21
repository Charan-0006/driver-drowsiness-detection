# realtime_demo_final.py 
import cv2, time, numpy as np, pickle
from pathlib import Path
import tensorflow as tf

try:
    from playsound import playsound
except Exception:
    playsound = None

# ---------- CONFIG ----------
EYE_MODEL = "weights/eye_model.h5"
YAWN_MODEL = "weights/mouth_model.h5"
HEAD_MODEL = "weights/head_model.h5"

FUSION_MODEL = "weights/fusion_model.h5"
SCALER = "weights/fusion_scaler.pkl"
ALARM = "alarm.mp3"                 # set None to disable sound
IMG_SIZE = (224,224)
WINDOW_S = 10
STRIDE_S = 5
THRESH = 0.5
CAMERA_ID = 0
# -----------------------------

def safe_load_model(path):
    if not path:
        return None
    p = Path(path)
    if p.exists():
        try:
            m = tf.keras.models.load_model(str(p))
            print(f"[OK] loaded model {p}")
            return m
        except Exception as e:
            print(f"[WARN] failed to load {p}: {e}")
            return None
    else:
        print(f"[INFO] model not found: {p} (skipping)")
        return None

def load_scaler(path):
    if not path:
        return None
    p = Path(path)
    if p.exists():
        try:
            with open(p, "rb") as f:
                s = pickle.load(f)
            print(f"[OK] loaded scaler {p}")
            return s
        except Exception as e:
            print(f"[WARN] failed to load scaler {p}: {e}")
            return None
    else:
        print(f"[INFO] scaler not found: {p} (skipping)")
        return None

def prep(img):
    try:
        r = cv2.resize(img, IMG_SIZE)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype("float32")/255.0
        return np.expand_dims(r, axis=0)
    except Exception:
        return None

def run_demo():
    eye = safe_load_model(EYE_MODEL)
    yawn = safe_load_model(YAWN_MODEL)
    head = safe_load_model(HEAD_MODEL)
    sh = safe_load_model(SHOULDER_MODEL) if SHOULDER_MODEL else None
    fusion = safe_load_model(FUSION_MODEL)
    scaler = load_scaler(SCALER)

    if fusion is None:
        print("[WARN] Fusion model not found. Fusion inference cannot run. Exiting.")
        return

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Exiting.")
        return

    buffer = []   # list of dicts {ts, prob_eye, prob_yawn, head_val, prob_shoulder}
    last_eval = 0.0
    print("[INFO] Starting realtime demo. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame from camera. Retrying...")
            time.sleep(0.1)
            continue

        ts = time.time()
        m = prep(frame)
        def safe_pred(mod):
            try:
                return float(mod.predict(m)[0,0]) if (mod is not None and m is not None) else 0.0
            except Exception:
                return 0.0

        p_eye = safe_pred(eye)
        p_yawn = safe_pred(yawn)
        p_head = safe_pred(head)
        p_sh = safe_pred(sh)

        buffer.append({"ts": ts, "prob_eye": p_eye, "prob_yawn": p_yawn, "head_val": p_head, "shoulder_val": p_sh})

        # drop old entries older than WINDOW_S*2 to keep buffer small
        cutoff = ts - WINDOW_S * 2
        buffer = [b for b in buffer if b["ts"] >= cutoff]

        # Evaluate every STRIDE_S seconds
        if ts - last_eval >= STRIDE_S:
            last_eval = ts
            sub = sorted(buffer, key=lambda x: x["ts"])
            perclos = np.mean([1 if s["prob_eye"]>0.5 else 0 for s in sub]) if sub else 0.0

            # longest continuous closed in sub
            max_closed = 0.0; seg_start = None; prev_ts = None
            for s in sub:
                if s["prob_eye"] > 0.5:
                    if seg_start is None: seg_start = s["ts"]
                    prev_ts = s["ts"]
                else:
                    if seg_start is not None and prev_ts is not None:
                        dur = prev_ts - seg_start
                        if dur > max_closed: max_closed = dur
                    seg_start = None; prev_ts = None
            if seg_start is not None and prev_ts is not None:
                dur = prev_ts - seg_start
                if dur > max_closed: max_closed = dur

            last_ts = sub[-1]["ts"] if sub else ts
            yawn_count = len([s for s in buffer if s["ts"] >= last_ts - 60.0 and s["prob_yawn"] > 0.5])
            avg_yawn = np.mean([s["prob_yawn"] for s in sub]) if sub else 0.0
            avg_head = np.mean([s["head_val"] for s in sub]) if sub else 0.0
            avg_sh = np.mean([s["shoulder_val"] for s in sub]) if sub else 0.0

            feat = np.array([[perclos, max_closed, yawn_count, avg_yawn, avg_head, avg_sh]])
            if scaler is not None:
                try:
                    feat_scaled = scaler.transform(feat)
                except Exception as e:
                    print("[WARN] scaler transform failed:", e); feat_scaled = feat
            else:
                feat_scaled = feat

            try:
                prob = float(fusion.predict(feat_scaled)[0,0])
            except Exception as e:
                print("[WARN] fusion predict failed:", e); prob = 0.0

            # overlay on frame
            label = f"DrowsyProb:{prob:.2f}"
            color = (0,255,0) if prob < THRESH else (0,0,255)
            if prob > THRESH:
                cv2.putText(frame, "DROWSINESS ALERT! WAKE UP!", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                print("[ALERT]", prob)
                if ALARM and playsound is not None:
                    try:
                        playsound(ALARM, block=False)
                    except Exception:
                        pass

            cv2.putText(frame, label, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()
    print("[INFO] Demo stopped.")

if __name__ == "__main__":
    run_demo()