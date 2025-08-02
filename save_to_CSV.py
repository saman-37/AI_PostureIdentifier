import os
import cv2
import mediapipe as mp
import pandas as pd
from feature_extractor import extract_features

video_folder = "Dataset"
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

data = []

for filename in os.listdir(video_folder):
    if filename.endswith(".mp4"):
        filepath = os.path.join(video_folder, filename)

        if filename.startswith("C"):
            label = 1
        elif filename.startswith("W"):
            label = 0
        else:
            continue

        cap = cv2.VideoCapture(filepath)
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
            features.append(label)
            data.append(features)

columns = [
    'shoulder_slope_start', 'knee_dist_start', 'knee_align_L_start', 'knee_align_R_start', 'shoulder_hip_angle_start',
    'shoulder_slope_bottom', 'knee_dist_bottom', 'knee_align_L_bottom', 'knee_align_R_bottom', 'shoulder_hip_angle_bottom',
    'label'
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("squat_features.csv", index=False)
print("✅ Features saved to 'squat_features.csv'")
