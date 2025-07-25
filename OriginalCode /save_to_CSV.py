import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from feature_extractor import extract_features  # Make sure this function is in Feature_extractor.py

# Path to video folder
video_folder = "../Dadaset"

#This reads only the first frame of each video.we must get 2 at least, or average multiple frames 
# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# List to collect data
data = []

# Loop through all files
for filename in os.listdir(video_folder):
    if filename.endswith(".mp4"):
        filepath = os.path.join(video_folder, filename)

        # Assign label based on filename
        if filename.startswith("C"):
            label = 1  # Correct posture
        elif filename.startswith("W"):
            label = 0  # Wrong posture
        else:
            continue  # Skip unknown file formats

        # Open video and read first frame
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert frame to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # If pose is detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            features = extract_features(landmarks)
            features.append(label)
            data.append(features)

        cap.release()

# Define column names
columns = ['shoulder_slope', 'knee_distance', 'knee_alignment', 'shoulder_alignment', 'label']

# Save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv("squat_features.csv", index=False)
