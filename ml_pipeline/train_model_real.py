import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# import tensorflowjs as tfjs # Removed to avoid protobuf version conflicts

print("1. Loading Features and Labels...")
X = np.load('X_features.npy')
labels = np.load('y_labels.npy')

with open('actions.json', 'r') as f:
    ACTIONS = json.load(f)

NUM_CLASSES = len(ACTIONS)
NUM_COORD_POINTS = 21 * 3 * 2 # 21 landmarks * 3 (x,y,z) * 2 hands = 126 features

y = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES).astype(int)

# Check if there is enough data
if len(X) < 10:
    print(f"Error: Not enough data points ({len(X)}).")
    exit(1)

# Split dataset perfectly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print("\n2. Designing the Deep Learning Neural Network Architecture...")
model = tf.keras.models.Sequential([
    # Input Layer matching MediaPipe's exact 126 coordinate output
    tf.keras.layers.Input(shape=(NUM_COORD_POINTS,)),
    
    # Hidden layers designed to extract geometric relationships between fingers
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Prevent overfitting
    tf.keras.layers.Dense(64, activation='relu'),
    
    # Output layer for all distinct gestures
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['categorical_accuracy']
)

print("\n3. Training the AI Model...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

print("\n4. Evaluating the AI Model Accuracy...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal AI Accuracy: {accuracy * 100:.2f}%")

print("\n5. Exporting Core Model to Keras and SavedModel format...")
model.save('isl_model.h5')
model.save('ml_pipeline/isl_saved_model')

# Also save the labels dictionary so the web app knows what class 0,1,2 means
export_path = r"C:\Users\admin\ISL Decoder\public\models\isl_model"
os.makedirs(export_path, exist_ok=True)
with open(os.path.join(export_path, "labels.json"), "w") as f:
    json.dump(ACTIONS, f)

print(f"\nAI Training Complete! Model saved to 'isl_model.h5' and 'ml_pipeline/isl_saved_model'")
print("Next step: Convert this model to TensorFlow.js format.")
