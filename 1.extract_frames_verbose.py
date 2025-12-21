# 1.extract_frames_verbose.py
import cv2
from pathlib import Path
import os

# EDIT: set to the EXTRACTED folder (not the .rar)
VIDEOS_ROOT = "E:\\YawDD.rar\\YawDD dataset"
FRAMES_OUT = "E:\\YawDD.rar\\YawDD_frames"
   # <<--- where frames will be saved
FPS_SAMPLE  = 5                      # frames per second to extract
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}

def extract_all(videos_root=VIDEOS_ROOT, out_root=FRAMES_OUT, fps_sample=FPS_SAMPLE):
    videos_root = Path(videos_root)
    out_root = Path(out_root)
    if not videos_root.exists():
    
        print("ERROR: videos_root does not exist:", videos_root)
        return
    out_root.mkdir(parents=True, exist_ok=True)

    # find all video files recursively
    video_files = [p for p in videos_root.rglob("*") if p.suffix in VIDEO_EXTS]
    print(f"Found {len(video_files)} video files under {videos_root}")

    total_saved = 0
    failed = []
    for vid in sorted(video_files):
        try:
            # preserve relative path inside out_root: e.g., dash/video1 -> out_root/dash/video1/
            rel = vid.relative_to(videos_root).with_suffix('')  # remove ext
            out_dir = out_root / rel.parent / rel.stem
            out_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(vid))
            if not cap.isOpened():
                print(f"[SKIP] Cannot open video: {vid}")
                failed.append((vid, "cap_not_open"))
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            interval = max(1, int(round(video_fps / float(fps_sample))))
            idx = 0
            saved = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % interval == 0:
                    ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    fn = f"{vid.stem}_frame{idx:06d}_t{ts_ms}ms.jpg"
                    cv2.imwrite(str(out_dir / fn), frame)
                    saved += 1
                idx += 1
            cap.release()
            total_saved += saved
            print(f"[OK] {vid} -> saved {saved} frames -> {out_dir}")
        except Exception as e:
            print(f"[ERROR] processing {vid}: {e}")
            failed.append((vid, str(e)))

    print("Done. total frames saved:", total_saved)
    if failed:
        print("Some videos failed:", len(failed))
        for v,reason in failed[:10]:
            print(" -", v, ":", reason)

if __name__ == "__main__":
    extract_all()
