# 2_auto_label.py
"""
Use MediaPipe to compute EAR, MAR, head pitch, shoulder slope for each frame,
and save per_frame_annotations.csv with computed values and binary labels.
"""
import mediapipe as mp
import cv2, math, os, re, csv
from pathlib import Path
import pandas as pd

# CONFIG - edit thresholds and paths
FRAMES_ROOT = r"E:\YawDD\YawDD_frames"

OUT_CSV = "per_frame_annotations.csv"
EAR_THRESH = 0.22
MAR_THRESH = 0.6
HEAD_PITCH_THRESH = 12.0  # degrees
SHOULDER_SLOPE_THRESH = 10.0  # degrees

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# indices chosen as reasonable defaults (MediaPipe FaceMesh)
LEFT_EYE_IDX = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX = 308
NOSE_IDX = 1
CHIN_IDX = 199
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

def euclid(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
def ear_from_points(points):
    A = euclid(points[1], points[5]); B = euclid(points[2], points[4]); C = euclid(points[0], points[3])
    return (A+B)/(2.0*C) if C>0 else 0.0

def mar_from_points(top,bottom,left,right):
    v = euclid(top,bottom); h = euclid(left,right); return v/h if h>0 else 0.0

def angle_deg(p1,p2):
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
    if dx==0: return 90.0 if dy>0 else -90.0
    return math.degrees(math.atan2(dy,dx))

def parse_ts(path):
    m = re.search(r"_t(\d+)ms", Path(path).stem)
    if m: return int(m.group(1))/1000.0
    return os.path.getmtime(path)

def analyze_image(img):
    h,w,_ = img.shape
    face_coords = {}
    pose_coords = {}
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        res = fm.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if res and res.multi_face_landmarks:
            for i,lm in enumerate(res.multi_face_landmarks[0].landmark):
                face_coords[i] = (int(lm.x*w), int(lm.y*h))
    with mp_pose.Pose(static_image_mode=True) as pm:
        res2 = pm.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        if res2 and res2.pose_landmarks:
            for i,lm in enumerate(res2.pose_landmarks.landmark):
                pose_coords[i] = (int(lm.x*w), int(lm.y*h))
    return face_coords, pose_coords

def process_all(frames_root=FRAMES_ROOT, out_csv=OUT_CSV):
    rows=[]
    for f in sorted(Path(frames_root).rglob("*.jpg")):
        img = cv2.imread(str(f))
        if img is None: continue
        face, pose = analyze_image(img)
        ts = parse_ts(f)
        ear=None; mar=None; pitch=None; shoulder_slope=None
        eye_closed=0; yawn=0; head_tilt=0; shoulder_slump=0
        try:
            if all(i in face for i in LEFT_EYE_IDX) and all(i in face for i in RIGHT_EYE_IDX):
                le=[face[i] for i in LEFT_EYE_IDX]; re=[face[i] for i in RIGHT_EYE_IDX]
                ear = (ear_from_points(le)+ear_from_points(re))/2.0
                if ear < EAR_THRESH: eye_closed=1
            if all(i in face for i in [MOUTH_TOP_IDX,MOUTH_BOTTOM_IDX,MOUTH_LEFT_IDX,MOUTH_RIGHT_IDX]):
                top=face[MOUTH_TOP_IDX]; bottom=face[MOUTH_BOTTOM_IDX]; left=face[MOUTH_LEFT_IDX]; right=face[MOUTH_RIGHT_IDX]
                mar = mar_from_points(top,bottom,left,right)
                if mar > MAR_THRESH: yawn=1
            if NOSE_IDX in face and CHIN_IDX in face:
                nose=face[NOSE_IDX]; chin=face[CHIN_IDX]
                dy = chin[1]-nose[1]; dx = chin[0]-nose[0]
                pitch = math.degrees(math.atan2(dy, max(1,abs(dx))))
                if abs(pitch) > HEAD_PITCH_THRESH: head_tilt=1
            if LEFT_SHOULDER_IDX in pose and RIGHT_SHOULDER_IDX in pose:
                ls=pose[LEFT_SHOULDER_IDX]; rs=pose[RIGHT_SHOULDER_IDX]
                slope = abs(angle_deg(ls,rs)); shoulder_slope = slope
                if shoulder_slope > SHOULDER_SLOPE_THRESH: shoulder_slump=1
        except Exception as e:
            print("frame error", f, e)
        rows.append({
            "frame": str(f),
            "ts_s": ts,
            "ear": ear,
            "mar": mar,
            "head_pitch": pitch,
            "shoulder_slope": shoulder_slope,
            "eye_closed": eye_closed,
            "yawn": yawn,
            "head_tilt": head_tilt,
            "shoulder_slump": shoulder_slump
        })
        if len(rows)%500==0: print("processed", len(rows))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved", out_csv)

if __name__ == "__main__":
    process_all()
