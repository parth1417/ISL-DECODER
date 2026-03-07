import numpy as np
import os
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
import tensorflow as tf

def convert():
    model_path = 'ml_pipeline/isl_model.h5'
    export_path = 'public/models/isl_model'
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Converting to TensorFlow.js format...")
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, export_path)
    print(f"Model successfully converted and saved to {export_path}")

if __name__ == "__main__":
    try:
        convert()
    except Exception as e:
        print(f"Conversion failed: {e}")
