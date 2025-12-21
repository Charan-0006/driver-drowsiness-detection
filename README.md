Driver Drowsiness Detection Using Deep Learning

This repository contains a real-time driver drowsiness detection system built using computer vision and deep learning.
The system identifies driver fatigue by analyzing eye closure, yawning, and head posture, and then applies temporal analysis to make reliable drowsiness decisions.

The project is designed to be non-intrusive, GPU-optimized, and suitable for real-time applications such as driver monitoring systems.

📌 Problem Statement

Driver drowsiness is one of the major causes of road accidents worldwide.
Fatigue develops gradually and is often unnoticed by drivers themselves.

This project aims to:

Detect early signs of driver drowsiness

Reduce accident risk using vision-based monitoring

Avoid intrusive sensors like EEG or wearable devices

🧠 Solution Overview

The system uses a multi-stage deep learning pipeline:

Facial video frames are extracted from driving videos

Facial landmarks are detected automatically

Behavioral cues are derived:

Eye closure

Yawning

Head-down posture

Separate CNN models analyze each cue

Temporal features are computed using sliding windows

A fusion model combines all cues for final drowsiness detection

A real-time webcam demo displays alerts when fatigue is detected

🏗️ Project Pipeline

Frame Extraction
Extract frames from input driving videos

Automatic Labeling
Use facial landmarks to compute:

Eye Aspect Ratio (EAR)

Mouth Aspect Ratio (MAR)

Head pitch angle
Labels are generated automatically (no manual annotation)

CNN Training

Eye-state CNN (open / closed)

Yawn detection CNN

Head posture CNN

GPU-Based Inference

Batch inference on video frames

Resume-capable processing

Mixed precision for speed

Temporal Feature Extraction

PERCLOS

Maximum eye-closure duration

Yawn frequency

Average head posture

Fusion Model

Combines temporal + spatial features

Produces final drowsiness prediction

Real-Time Detection

Webcam-based monitoring

Visual alerts for drowsiness

📁 Repository Structure
Code files/
├── 1.extract_frames_verbose.py
├── 2_auto_label.py
├── 3_prepare_datasets.py
├── 4_train_eye_cnn.py
├── 5_train_yawn_cnn.py
├── 6_train_head_cnn.py
├── 7_run_inference_and_extract_features.py
├── 8_train_fusion.py
├── 9_realtime_demo_final.py

📄 Description of Files
File Name	Purpose
1.extract_frames_verbose.py	Extracts frames from driving videos
2_auto_label.py	Automatically labels eye, mouth, and head posture using landmarks
3_prepare_datasets.py	Prepares datasets for CNN training
4_train_eye_cnn.py	Trains CNN for eye open/closed detection
5_train_yawn_cnn.py	Trains CNN for yawning detection
6_train_head_cnn.py	Trains CNN for head posture detection
7_run_inference_and_extract_features.py	GPU-optimized inference + temporal feature extraction
8_train_fusion.py	Trains fusion model using temporal features
9_realtime_demo_final.py	Real-time webcam drowsiness detection


📊 Dataset Used
YawDD (Yawning Detection Dataset)

Publicly available dataset for driver fatigue research

Contains real driving videos

Includes frontal and side camera views

Suitable for yawning and drowsiness analysis

🔗 Dataset link:
https://www.site.uottawa.ca/~shervin/yawdd/

⚠️ Dataset is NOT included in this repository
Users must download it separately due to size constraints.

🛠️ Technologies & Tools

Programming Language: Python

Deep Learning: TensorFlow / Keras

Computer Vision: OpenCV

Facial Landmarks: MediaPipe

Numerical Computing: NumPy, Pandas

Visualization: Matplotlib

Hardware Acceleration: GPU (CUDA supported)

📦 Requirements

Create a virtual environment (recommended), then install:

opencv-python
mediapipe
tensorflow
numpy
pandas
tqdm
matplotlib
scikit-learn

⚙️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection

2️⃣ Prepare Dataset

Download YawDD dataset

Extract videos

Update dataset paths inside scripts

3️⃣ Train Models (Optional)

Run scripts in order:

python 4_train_eye_cnn.py
python 5_train_yawn_cnn.py
python 6_train_head_cnn.py
python 8_train_fusion.py


Pretrained weights are not included in this repository.

4️⃣ Run Real-Time Detection
python 9_realtime_demo_final.py


⚠️ Webcam access works only on local machine (not on GitHub).

🎥 Real-Time Demo

A real-time webcam demo shows:

Eye closure detection

Yawning detection

Head posture analysis

Drowsiness alert generation

📌 Demo video: (Add YouTube / Google Drive link here)

🚫 What Is Not Included

Dataset files

Trained model weights (.h5)

Generated CSV files

Videos or large media files

These are excluded intentionally to keep the repository lightweight.

🎯 Applications

Driver monitoring systems

Intelligent transportation systems

Fleet safety solutions

Research in fatigue detection

Advanced Driver Assistance Systems (ADAS)

👨‍💻 Authors
Sri charan N , B uday ,Vishnu P
B.Tech – Cyber Security
Amrita School of Computing, Amritapuri

📄 License
This project is released under the MIT License.
