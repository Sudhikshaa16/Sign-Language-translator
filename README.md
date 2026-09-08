# Bidirectional Real-Time Sign Language Translation System

A lightweight sign language recognition prototype built with Python, OpenCV, MediaPipe, and PyTorch. The system detects hand landmarks from webcam input and classifies predefined gestures into text. It also includes a simple text-to-sign visualization using saved landmark data.

## Features

* Real-time sign-to-text recognition
* MediaPipe hand landmark detection
* PyTorch MLP gesture classification
* Custom gesture data collection and training
* Text-to-sign visualization
* Streamlit-based interface

## Supported Gestures

```text
HELLO
YES
NO
PLEASE
THANKYOU

```

## Tech Stack

Python • OpenCV • MediaPipe • PyTorch • Streamlit • NumPy • Matplotlib • scikit-learn

## Architecture

```text
Webcam
  ↓
MediaPipe Hand Landmarks
  ↓
Feature Extraction
  ↓
PyTorch MLP
  ↓
Gesture Prediction
  ↓
Text Output
```

## How It Works

1. Captures hand gestures through the webcam.
2. Extracts 21 hand landmarks using MediaPipe.
3. Converts landmarks into a feature vector.
4. Predicts the gesture using a trained PyTorch MLP.
5. Displays the predicted gesture in the Streamlit application.

## Project Structure

```text
sign_language_translator/
├── app/
│   ├── main.py
│   ├── sign_to_text.py
│   └── text_to_sign.py
├── model/
│   ├── __init__.py
│   ├── dataset.py
│   ├── gesture_dataset.py
│   ├── gesture_mlp.pth
│   ├── inference.py
│   ├── mlp.py
│   ├── model_lstm.py
│   ├── model_transformer.py
│   ├── train.py
│   └── trained_model.pth
├── scripts/
│   ├── augmenter.py
│   ├── dataset_generator.py
│   ├── record_real_data.py
│   ├── record_streamlit.py
│   ├── visualize.py
│   └── visualizer.py
├── dataset/
│   ├── label.txt
│   ├── raw/
│   ├── real/
│   └── synthetic/
├── configs/
│   └── train_config.yaml
├── assets/
│   ├── icons/
│   └── signs_videos/
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/main.py
```

## Train Model

```bash
python -m model.train
```

## Dataset

Custom landmark samples are stored in:

```text
dataset/
├── real/
├── raw/
└── synthetic/
```

## Limitations

This project is a prototype with a limited gesture vocabulary. Performance can vary depending on lighting, camera quality, hand position, and dataset diversity.

