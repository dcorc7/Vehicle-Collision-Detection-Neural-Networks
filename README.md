# Deep Learning for Collision Detection from Dashcam Footage


## Introduction

This project addresses the challenge of real-time collision detection and classification using dashcam video footage, a task with critical implications for road safety, autonomous systems, and insurance verification. We applied deep learning models to classify video clips as containing collisions/near-misses or no risk of collision at all. Our approach relies on time-series features extracted from video frames using YOLO-based object detection and processes them through sequence modeling architectures to detect temporal patterns indicative of road incidents. 


## Team

Project Group 11

Member 1 – David Corcoran

Member 2 – Sean Morris

Member 3 – Adam Stein

Member 4 – Hung Tran


## Project Structure

```plaintext
├── README.md                        # Project overview and setup instructions.
├── code/                            # Source notebooks and scripts for modeling and processing.
│   ├── data_processing.ipynb        # OExtract frames from each  video.
│   ├── yolo_detection.ipynb         # Applying YOLOv8 for frame-wise object detection.
│   ├── yolov8n.pt                   # Pretrained YOLOv8n weights used for object detection.
│   ├── GRU/                         # GRU model training, evaluation, and metrics.
│   ├── LSTM/                        # LSTM model training, evaluation, and hyperparameter tuning.
│   └── Transformer/                 # Transformer-based sequence model for temporal classification.
├── data/                            # Dashcam video data (external link due to file size).
└── Summary-Write-Up.pdf             # Final technical report with methodology, results, and discussion.
```

## Data
The data/ directory is used to store both raw video datasets and preprocessed feature sequences. Due to storage limitations, the actual dashcam footage is not included in this repository.

Data processing steps include:

Extracting YOLOv8 detections from each frame.

Deriving motion-based features from bounding box sequences.

Formatting data for input into temporal neural network architectures.

You can download the full dataset and processed features from [this link](https://drive.google.com/uc?export=download&id=1c_S2MOFZySFJPpd8GTKxSXKv7SVbG0ad)

=======
# Vehicle-Collision-Detection-Neural-Networks
