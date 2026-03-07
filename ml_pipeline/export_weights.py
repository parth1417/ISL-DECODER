import numpy as np
import tensorflow as tf
import json
import os

def export_weights():
    model_path = 'isl_model.h5'
    if not os.path.exists(model_path):
        print("Model not found")
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
    
    with open('public/models/isl_model/weights.json', 'w') as f:
        json.dump(weights_data, f)
    
    print("Exported weights to public/models/isl_model/weights.json")

if __name__ == "__main__":
    export_weights()
