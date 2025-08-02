import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from feature_extractor import extract_features

# 1. Load the model
with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

# 2. Set video path
video_path = os.path.join(os.getcwd(), "TestSet", "T2.mov")
print("✅ Video exists:", os.path.exists(video_path))

# 3. Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# 4. Process the video
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Find top and bottom frames
top_frame_idx = None
bottom_frame_idx = None
highest_hip_y = 1.0
lowest_hip_y = 0.0

for i in range(0, frame_count, 5):  # Sample every 5 frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y

        if hip_y < highest_hip_y:
            highest_hip_y = hip_y
            top_frame_idx = i
        if hip_y > lowest_hip_y:
            lowest_hip_y = hip_y
            bottom_frame_idx = i

# 5. Extract features (select 5 from both top and bottom frames)
features = []
for idx in [top_frame_idx, bottom_frame_idx]:
    if idx is None:
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        feats = extract_features(landmarks)
        features.extend(feats[:5])  # Select only 5 features from each frame
        print(f"✅ Features extracted from frame {idx} (count: {len(feats)})")

cap.release()

# 6. Predict
if len(features) >= 5:  # Ensure at least 5 features
    # Use only the first 5 features (from the top frame)
    prediction = knn.predict([features[:5]])[0]  
    print("Final prediction:", "Correct posture" if prediction == 1 else "Incorrect posture")
else:
    print(f"❌ Not enough features: {len(features)} (required: 5)")
