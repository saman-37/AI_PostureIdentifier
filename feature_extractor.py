import numpy as np
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_features(landmarks):
    """
    Extract features using only knees and shoulders
    """
    features = []
    
    # Extract joint coordinates (condensed into one line)
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    
    # 1. Shoulder tilt angle (relative to horizontal line)
    shoulder_slope = np.arctan2(right_shoulder[1] - left_shoulder[1], 
                                right_shoulder[0] - left_shoulder[0])
    features.append(np.degrees(shoulder_slope))
    
    # 2. Distance between knees
    knee_distance = np.sqrt((right_knee[0] - left_knee[0])**2 + (right_knee[1] - left_knee[1])**2)
    features.append(knee_distance)
    
    # 3. Vertical alignment of knees to shoulders
    knee_alignment_left = left_knee[0] - left_shoulder[0]
    knee_alignment_right = right_shoulder[0] - right_knee[0]
    features.extend([knee_alignment_left, knee_alignment_right])
    
    return features
