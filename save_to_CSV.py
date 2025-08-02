import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from feature_extractor import extract_features

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# List to store the data
data = []

video_folder = "Dataset"  # Relative path → refers to 'Dataset' folder in current directory

for filename in sorted(os.listdir(video_folder)):  # Process files in alphabetical order
    if not (filename.startswith("C") or filename.startswith("W")) or not filename.endswith(".mp4"):
        continue  # Only process MP4 files starting with C or W

    label = 1 if filename.startswith("C") else 0
    cap = cv2.VideoCapture(os.path.join(video_folder, filename))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 5th frame (100% → 20% sampling)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 5 == 0:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                try:
                    features = extract_features(results.pose_landmarks.landmark)
                    data.append(features + [label])
                except:
                    continue

    cap.release()

# Save to CSV
if data:
    columns = [
        'shoulder_slope', 'knee_distance', 'knee_align_L', 'knee_align_R', 'shoulder_hip_angle',
        # ... (Add all other feature names returned by feature_extractor.py here)
        'label'
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("squat_features.csv", index=False)
    print(f"\nFinal number of saved samples: {len(df)}")
    print("Class distribution:\n", df['label'].value_counts())
else:
    print("❌ No data extracted. Please check the videos or pose detection.")
