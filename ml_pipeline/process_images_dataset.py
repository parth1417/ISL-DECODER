import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import random

mp_hands = mp.solutions.hands

RAW_VIDEOS_PATH = r'C:\Users\admin\Downloads\Indian'

def extract_keypoints(hand_landmarks):
    if hand_landmarks:
        return np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
    return np.zeros(21*3)

def process_images(max_images_per_class=100):
    if not os.path.exists(RAW_VIDEOS_PATH):
        print(f"Error: {RAW_VIDEOS_PATH} not found.", flush=True)
        return

    all_items = os.listdir(RAW_VIDEOS_PATH)
    actions = sorted([d for d in all_items if os.path.isdir(os.path.join(RAW_VIDEOS_PATH, d)) and len(d) <= 2])
    if len(actions) == 0:
        # Fallback if classes have longer names
        actions = sorted([d for d in all_items if os.path.isdir(os.path.join(RAW_VIDEOS_PATH, d)) and d not in ['isl_saved_model', 'raw_videos']])
    
    if len(actions) == 0:
        print(f"No valid class directories found in {RAW_VIDEOS_PATH}/", flush=True)
        return

    features = []
    labels = []
    
    # Save the action names to a JSON file so train_model can read them
    import json
    with open('actions.json', 'w') as f:
        json.dump(actions, f)

    print(f"Found {len(actions)} classes. Processing up to {max_images_per_class} images per class...", flush=True)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        for label_idx, action in enumerate(actions):
            action_path = os.path.join(RAW_VIDEOS_PATH, action)
            if not os.path.isdir(action_path):
                continue
            
            images = glob.glob(os.path.join(action_path, '*.jpg'))
            random.shuffle(images)
            images_to_process = images[:max_images_per_class]
            
            successful_extracts = 0
            for img_path in images_to_process:
                try:
                    image = cv2.imread(img_path)
                    if image is None: continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)
                    
                    # MediaPipe Hands returns multi_hand_landmarks and multi_handedness
                    if results.multi_hand_landmarks:
                        # We need to map which hand is left and which is right
                        # If only one hand is detected, we need to fill the other with zeros
                        lh_keypoints = np.zeros(21*3)
                        rh_keypoints = np.zeros(21*3)
                        
                        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            handedness = results.multi_handedness[idx].classification[0].label
                            if handedness == 'Left':
                                lh_keypoints = extract_keypoints(hand_landmarks)
                            else:
                                rh_keypoints = extract_keypoints(hand_landmarks)
                        
                        keypoints = np.concatenate([lh_keypoints, rh_keypoints])
                        features.append(keypoints)
                        labels.append(label_idx)
                        successful_extracts += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}", flush=True)
            
            print(f"Processed '{action}': {successful_extracts} valid hand gestures from {len(images_to_process)} images", flush=True)

    np.save('X_features.npy', np.array(features))
    np.save('y_labels.npy', np.array(labels))
    print(f"Done! Saved {len(features)} total features to 'X_features.npy' and 'y_labels.npy'.", flush=True)

if __name__ == "__main__":
    process_images(max_images_per_class=100)
