import numpy as np
import json
import os

def export_for_js():
    X = np.load('X_features.npy')
    y = np.load('y_labels.npy')
    
    # Save as JSON
    data = {
        'X': X.tolist(),
        'y': y.tolist()
    }
    
    with open('ml_pipeline/dataset_for_js.json', 'w') as f:
        json.dump(data, f)
    
    print("Exported dataset to ml_pipeline/dataset_for_js.json")

if __name__ == "__main__":
    export_for_js()
