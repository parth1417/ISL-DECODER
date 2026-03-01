# ISL Dataset Processor 
This folder contains the tools to download and prepare the INCLUDE Indian Sign Language (ISL) dataset for deep learning.

## Setup Requirements

1. **Install Python**: Make sure you have Python 3.8+ installed on your computer.
2. **Install Dependencies**: Open a terminal in this `ml_pipeline` folder and run:
   ```bash
   pip install -r requirements.txt
   ```

## How to use

1. Download the INCLUDE dataset videos (or any `.mp4` sign language videos) and place them in structured folders inside a `raw_videos` directory.
   Example structure:
   ```
   raw_videos/
   ├── Hello/
   │   ├── video1.mp4
   │   └── video2.mp4
   ├── ThankYou/
   │   ├── video1.mp4
   │   └── video2.mp4
   ```

2. Run the processing script to extract the 3D MediaPipe coordinates from every frame:
   ```bash
   python process_dataset.py
   ```

3. The script will generate a new folder called `processed_data/` containing `.npy` (NumPy) files. These files represent pure mathematical coordinate sequences of the hand gestures, completely removing the background, lighting, and people, which allows the AI to learn 100x faster!
