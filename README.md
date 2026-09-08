# Sign Language Translator (Lightweight)

## What it does
Real-time webcam-based sign recognition (hand gestures) → text display, and text → sign animation (video playback of short sign clips). Designed to work with a small synthetic dataset generated from a few webcam recordings.

## Quick setup
1. Create and activate venv
2. `pip install -r requirements.txt`
3. Run `python scripts/dataset_generator.py --record` to collect a few seed samples.
4. Run `python scripts/augmenter.py` to generate synthetic dataset (saved to dataset/raw and dataset/synthetic).
5. Train: `python model/train.py --cfg configs/train_config.yaml`
6. Run app: `streamlit run app/main.py`

## File structure
(See the project root in repo)

## Notes
- Uses MediaPipe Hands landmarks (21 points per hand). Format: sequence of (T x 42) floats (x,y for 21 points). If both hands are present we will use more channels; code handles single-hand common use-case.
