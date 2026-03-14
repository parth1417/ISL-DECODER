import numpy as np
import os
import glob
import json

DATA_PATH = r'C:\Users\admin\Downloads\custom_isl_dataset\MP_Data'

# The hands are at indices 1536:1662 in the 1662 array
def process_dynamic_to_static():
    actions = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    
    with open('actions.json', 'w') as f:
        json.dump(actions, f)
        
    print(f"Found {len(actions)} classes: {actions}")
    
    features = []
    labels = []
    
    for label_idx, action in enumerate(actions):
        action_path = os.path.join(DATA_PATH, action)
        
        # Iterating through all sequence folders
        for sequence in os.listdir(action_path):
            seq_path = os.path.join(action_path, sequence)
            if not os.path.isdir(seq_path): continue
                
            # Iterate through all 30 frames
            for frame_file in glob.glob(os.path.join(seq_path, '*.npy')):
                try:
                    res = np.load(frame_file)
                    # Extract left and right hands (indices 1536 to 1662)
                    lh = res[1536:1599]
                    rh = res[1599:1662]
                    
                    # only use frame if at least one hand is detected
                    if np.sum(lh) != 0 or np.sum(rh) != 0:
                        keypoints = np.concatenate([lh, rh])
                        features.append(keypoints)
                        labels.append(label_idx)
                except Exception as e:
                    print(f"Error reading {frame_file}: {e}")

    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Final dataset shape: Features {features.shape}, Labels {labels.shape}")
    
    np.save('X_features.npy', features)
    np.save('y_labels.npy', labels)
    print("Saved extracted hands to X_features.npy and y_labels.npy")

if __name__ == "__main__":
    process_dynamic_to_static()
