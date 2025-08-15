# Image-to-Audio Caption Generator

A tiny multimodal data generator: drop images into `images/`, run `main.py`, and get:
- **Image captions** (BLIP) saved to `output/captions.csv`
- **Spoken audio files** (offline TTS via `pyttsx3`) in `output/audio/`
- **Normalized WAV** output for consistent quality

This simulates the “view → describe → record” workflow used in multimodal AI data collection.

## Why this project?
- Mirrors real tasks from AI research data roles: describing visuals, producing clean audio, and keeping structured datasets.
- Fully offline TTS (no API keys).
- Clear, recruiter-friendly structure.

## Quickstart

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .ita
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .ita/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Add a few images
mkdir -p images
# Put some .jpg/.png in the images/ folder

# 4) Run
python main.py
