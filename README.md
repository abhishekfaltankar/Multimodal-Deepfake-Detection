# Multimodal Deepfake Detection

This is my Final Year B.Tech (Artificial Intelligence & Data Science) project. The objective of this project is to detect whether a video is real or deepfake using deep learning. The project follows a complete pipeline from dataset preparation and preprocessing to model training and deployment.

---

## Project Overview

The project consists of the following stages:

- Video dataset preparation
- Frame extraction from videos
- Face detection and cropping
- Dataset creation for deep learning
- Transfer learning using EfficientNet-B0
- Model training and evaluation
- Streamlit web application for prediction

---

## Current Progress

### Phase 1 - Project Setup
- Created project structure
- Configured Git and GitHub repository
- Created Python 3.11 virtual environment
- Installed all required dependencies

### Phase 2 - Dataset Preparation
- Downloaded Celeb-DF dataset
- Created a sample dataset for development
- Organized Real and Fake video folders

### Phase 3 - Frame Extraction
- Extracted frames from sample videos
- Stored extracted frames in a structured directory

### Phase 4 - Face Detection
- Integrated OpenCV YuNet Face Detector
- Detected faces from extracted frames
- Cropped detected faces
- Resized all face images to 224 × 224
- Automatically generated processed face dataset

### Phase 5 - Deep Learning Pipeline
- Created a custom PyTorch Dataset
- Loaded all processed face images automatically
- Assigned labels for Real and Fake images
- Implemented DataLoader
- Split dataset into Training, Validation and Testing sets
- Integrated pretrained EfficientNet-B0
- Modified classifier for binary classification (Real / Fake)
- Successfully tested the model using dummy input

---

## Dataset

Dataset Used:

- Celeb-DF v2

Current Dataset:

- 20 Sample Videos
- 849 Cropped Face Images

Dataset Split:

- Training Images: 594
- Validation Images: 127
- Testing Images: 128

---

## Project Structure

```text
Multimodal-Deepfake-Detection/
│
├── app/
│
├── dataset/
│   ├── raw/
│   ├── sample/
│   └── processed/
│       ├── frames/
│       ├── faces/
│       └── audio/
│
├── models/
│   ├── efficientnet.py
│   └── face_detection_yunet_2023mar.onnx
│
├── preprocessing/
│   ├── extract_frames.py
│   └── detect_faces.py
│
├── training/
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── reports/
├── utils/
│
├── config.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python 3.11
- PyTorch
- Torchvision
- OpenCV
- YuNet Face Detector
- NumPy
- Pandas
- Pillow
- Streamlit

---

## Current Workflow

```text
Video Dataset
      │
      ▼
Frame Extraction
      │
      ▼
Face Detection
      │
      ▼
Face Cropping
      │
      ▼
Image Preprocessing
      │
      ▼
PyTorch Dataset
      │
      ▼
DataLoader
      │
      ▼
EfficientNet-B0
      │
      ▼
Model Training
      │
      ▼
Prediction
      │
      ▼
Streamlit Web Application
```

---

## Setup

Clone the repository

```bash
git clone https://github.com/abhishekfaltankar/Multimodal-Deepfake-Detection.git
```

Move into the project

```bash
cd Multimodal-Deepfake-Detection
```

Create virtual environment

```bash
py -3.11 -m venv venv
```

Activate virtual environment

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Completed

- [x] Project setup
- [x] Dataset preparation
- [x] Frame extraction
- [x] Face detection
- [x] Face cropping
- [x] Image preprocessing
- [x] Custom PyTorch Dataset
- [x] DataLoader
- [x] EfficientNet-B0 integration

---

## Remaining Work

- [ ] Model training
- [ ] Validation and testing
- [ ] Performance evaluation
- [ ] Model saving
- [ ] Prediction module
- [ ] Streamlit web application
- [ ] Final documentation

---

This repository contains the complete development process of our final year project, starting from dataset preparation to building a deep learning-based deepfake detection system using transfer learning.