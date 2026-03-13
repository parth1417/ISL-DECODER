import numpy as np
import tensorflow as tf
import json
import os

def export_weights():
    model_path = r'C:\Users\admin\ISL Decoder\isl_model.h5'
    if not os.path.exists(model_path):
        print("Model not found at", model_path)
        return

    model = tf.keras.models.load_model(model_path)
    
    weights_data = []
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            # Dense layers have [weights, biases]
            weights_data.append({
                'name': layer.name,
                'weights': weights[0].tolist(),
                'biases': weights[1].tolist()
            })
    
    out_dir = r'C:\Users\admin\ISL Decoder\public\models\isl_model'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'weights.json')
    
    with open(out_path, 'w') as f:
        json.dump(weights_data, f)
    
    print("Exported weights to", out_path)

if __name__ == "__main__":
    export_weights()
