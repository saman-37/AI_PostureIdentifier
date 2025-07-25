
import cv2
import mediapipe as mp
import numpy as np
import pickle
from feature_extractor import extract_features

with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

video_path = "../Dataset/test1.mp4"
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

top_frame_idx = None
bottom_frame_idx = None
highest_hip_y = 1.0
lowest_hip_y = 0.0

for i in range(0, frame_count, 5):
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

features = []
for idx in [top_frame_idx, bottom_frame_idx]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        continue
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        feats = extract_features(landmarks)
        features.extend(feats)
cap.release()

if len(features) == 10:
    prediction = knn.predict([features])[0]
    print("✅ Prediction:", "Correct posture" if prediction == 1 else "Incorrect posture")
else:
    print("❌ Not enough features to make prediction.")
