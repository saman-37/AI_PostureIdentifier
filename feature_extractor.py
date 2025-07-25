
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_features(landmarks):
    features = []

    # Extract key landmarks
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

    # Shoulder slope (angle)
    shoulder_slope = np.arctan2(right_shoulder[1] - left_shoulder[1],
                                right_shoulder[0] - left_shoulder[0])
    features.append(np.degrees(shoulder_slope))

    # Distance between knees
    knee_distance = np.sqrt((right_knee[0] - left_knee[0])**2 + (right_knee[1] - left_knee[1])**2)
    features.append(knee_distance)

    # Knee to shoulder alignment (horizontal distance)
    knee_align_L = left_knee[0] - left_shoulder[0]
    knee_align_R = right_shoulder[0] - right_knee[0]
    features.extend([knee_align_L, knee_align_R])

    # Shoulder-hip angle
    vec = [left_shoulder[0] - left_hip[0], left_shoulder[1] - left_hip[1]]
    vertical = [0, -1]
    dot = np.dot(vec, vertical)
    angle_rad = np.arccos(dot / (np.linalg.norm(vec) * np.linalg.norm(vertical)))
    features.append(np.degrees(angle_rad))

    return features
