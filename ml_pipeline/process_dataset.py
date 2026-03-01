import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Holistic (Tracks Face, Both Hands, and Body Pose)
mp_holistic = mp.solutions.holistic

# Paths
DATA_PATH = os.path.join('processed_data') 
RAW_VIDEOS_PATH = os.path.join('raw_videos')

# Function to extract math coordinates of hands
def extract_keypoints(results):
    # Flatten the 3D coordinates (x, y, z) for 21 points on each hand into numpy arrays
    # If a hand isn't visible in the frame, we fill it with zeros so the math model doesn't break
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Optional: You could also include face or pose landmarks for better ISL context mapping in the future
    return np.concatenate([lh, rh])

def process_videos():
    if not os.path.exists(RAW_VIDEOS_PATH):
        print(f"Error: Please create a '{RAW_VIDEOS_PATH}' folder and add your ISL video folders inside.")
        os.makedirs(RAW_VIDEOS_PATH)
        return

    # Loop through each folder (Action/Sign) in raw_videos
    actions = np.array(os.listdir(RAW_VIDEOS_PATH))
    
    if len(actions) == 0:
        print(f"No sign language video folders found in {RAW_VIDEOS_PATH}/")
        return

    # Create MediaPipe Holistic context
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            action_path = os.path.join(RAW_VIDEOS_PATH, action)
            if not os.path.isdir(action_path):
                continue
            
            videos = os.listdir(action_path)
            print(f"\nProcessing sign: {action} ({len(videos)} videos)")
            
            for video_num, video_name in enumerate(videos):
                video_path = os.path.join(action_path, video_name)
                cap = cv2.VideoCapture(video_path)
                
                # To standardize length for an LSTM Deep Learning model, 
                # we usually want a fixed number of frames per sequence (e.g., 30 frames per gesture)
                sequence_keypoints = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break # Video finished
                    
                    # MediaPipe needs color converted from BGR (OpenCV default) to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR@RGB)
                    image.flags.writeable = False # Speeds up memory processing
                    
                    # Neural Network processing! Get hand landmarks.
                    results = holistic.process(image)
                    
                    # Extract the pure coordinate math
                    keypoints = extract_keypoints(results)
                    sequence_keypoints.append(keypoints)

                # Standardize padding (pad with zeros or cut to exactly 30 frames)
                SEQUENCE_LENGTH = 30
                if len(sequence_keypoints) > SEQUENCE_LENGTH:
                    sequence_keypoints = sequence_keypoints[:SEQUENCE_LENGTH] # Truncate 
                else:
                    # Pad short videos
                    padding = [np.zeros(21*3*2)] * (SEQUENCE_LENGTH - len(sequence_keypoints))
                    sequence_keypoints.extend(padding)

                # Save the processed math to Numpy arrays!
                save_dir = os.path.join(DATA_PATH, action)
                os.makedirs(save_dir, exist_ok=True)
                
                npy_path = os.path.join(save_dir, f"{str(video_num)}.npy")
                np.save(npy_path, np.array(sequence_keypoints))
                
                print(f"Saved -> {npy_path}")
                
                cap.release()

    print("\n✅ Dataset Processing Complete!")
    print(f"Your raw videos have been converted into clean mathematical arrays inside the '{DATA_PATH}/' folder.")
    print("These .npy files are ready to be fed instantly into an LSTM Model for fast training.")

if __name__ == "__main__":
    process_videos()
