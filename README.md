# Multimodal Deepfake Detection

A Final Year B.Tech (AI & DS) project focused on detecting deepfake videos using deep learning. This project aims to analyze both visual and audio information to classify videos as real or fake.

## Project Objective

The goal of this project is to build a deepfake detection system by:

- Extracting frames from videos
- Detecting and cropping faces
- Training a deep learning model on face images
- Building a Streamlit application for prediction

---

## Current Progress

### ✅ Phase 1: Project Setup
- Created project structure
- Configured Git and GitHub repository
- Organized dataset folders

### ✅ Phase 2: Dataset Preparation
- Downloaded Celeb-DF dataset
- Created a sample dataset for testing
- Organized Real and Fake videos

### ✅ Phase 3: Frame Extraction
- Extracted frames from sample videos
- Stored frames in a structured directory

### ✅ Phase 4: Face Detection
- Switched to Python 3.11 virtual environment for compatibility
- Installed OpenCV 4.10
- Integrated OpenCV YuNet Face Detector
- Successfully detected faces from extracted frames
- Verified face detection on a sample image

---

## Project Structure

```
Multimodal-Deepfake-Detection/
│
├── app/
├── dataset/
│   ├── raw/
│   ├── sample/
│   └── processed/
│       ├── audio/
│       ├── frames/
│       └── faces/
│
├── models/
│
├── outputs/
│
├── preprocessing/
│   ├── extract_frames.py
│   └── detect_faces.py
│
├── training/
├── utils/
│
├── reports/
│
├── config.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python 3.11
- OpenCV
- YuNet Face Detector
- PyTorch (Upcoming)
- Streamlit (Upcoming)

---

## Current Workflow

```
Videos
   │
   ▼
Frame Extraction ✅
   │
   ▼
Face Detection ✅
   │
   ▼
Face Cropping (Next)
   │
   ▼
Dataset Preparation
   │
   ▼
Deep Learning Model Training
   │
   ▼
Evaluation
   │
   ▼
Streamlit Web Application
```

---

## Virtual Environment

Create virtual environment

```bash
py -3.11 -m venv venv
```

Activate (Windows)

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Completed

- [x] Project structure
- [x] Dataset organization
- [x] Sample dataset creation
- [x] Frame extraction
- [x] Python 3.11 virtual environment
- [x] OpenCV installation
- [x] YuNet model integration
- [x] Face detection on sample frame

## Upcoming

- [ ] Face extraction for entire dataset
- [ ] Image preprocessing
- [ ] Deep learning model training
- [ ] Model evaluation
- [ ] Streamlit application
- [ ] Final testing

---

This repository documents the development of my final year deepfake detection project from dataset preparation to model deployment.
