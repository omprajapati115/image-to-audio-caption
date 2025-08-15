import os
import csv
import hashlib
from pathlib import Path
from datetime import datetime

from PIL import Image
from tqdm import tqdm

import pyttsx3
from pydub import AudioSegment
from pydub.effects import normalize

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "output"
AUDIO_DIR = OUTPUT_DIR / "audio"
CAPTION_CSV = OUTPUT_DIR / "captions.csv"

OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# -----------------------------
# Captioning model (BLIP)
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
model.eval()

# -----------------------------
# Text-to-speech (offline)
# -----------------------------
def init_tts(rate=170, volume=1.0, voice_index=None):
    engine = pyttsx3.init()
    # Tune for clarity and natural pacing for 2–3 min sets later
    engine.setProperty('rate', rate)       # words per minute
    engine.setProperty('volume', volume)   # 0.0 to 1.0
    if voice_index is not None:
        voices = engine.getProperty('voices')
        if 0 <= voice_index < len(voices):
            engine.setProperty('voice', voices[voice_index].id)
    return engine

tts_engine = init_tts()

def tts_to_wav(text: str, wav_path: Path):
    # pyttsx3 can save directly to file via runAndWait with a temporary change
    tts_engine.save_to_file(text, str(wav_path))
    tts_engine.runAndWait()


# -----------------------------
# Utilities
# -----------------------------
def safe_stem(name: str) -> str:
    stem = Path(name).stem
    # keep-alnum-dash
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in stem)
    # add short hash to avoid collisions
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{keep}-{h}"

def load_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts:
            yield p

def generate_caption(img_path: Path, max_new_tokens=30, min_length=5):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    caption = processor.decode(out[0], skip_special_tokens=True).strip()

    # Light post-processing for readability
    if len(caption) < min_length:
        caption = f"An image likely showing {caption}".strip()

    # Capitalize and ensure ending period
    if caption and caption[0].islower():
        caption = caption[0].upper() + caption[1:]
    if caption and caption[-1].isalnum():
        caption += "."

    return caption

def normalize_wav(in_wav: Path, out_wav: Path, target_dbfs=-16.0):
    audio = AudioSegment.from_file(in_wav)
    # normalize to consistent loudness
    audio = normalize(audio)
    # Hard-limit target loudness if needed
    change = target_dbfs - audio.dBFS
    audio = audio.apply_gain(change)
    audio.export(out_wav, format="wav")

def ensure_csv_header(csv_path: Path):
    header = ["image_file", "caption", "audio_file", "timestamp"]
    write_header = not csv_path.exists()
    f = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    return f, writer

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    images = list(load_images(IMAGES_DIR))
    if not images:
        print(f"[INFO] Put some images into: {IMAGES_DIR} and run again.")
        return

    f, writer = ensure_csv_header(CAPTION_CSV)

    try:
        for img_path in tqdm(images, desc="Processing images"):
            try:
                stem = safe_stem(img_path.name)

                # 1) Caption
                caption = generate_caption(img_path)

                # 2) TTS → WAV (raw)
                raw_wav = AUDIO_DIR / f"{stem}-raw.wav"
                tts_to_wav(caption, raw_wav)

                # 3) Normalize → WAV (final)
                final_wav = AUDIO_DIR / f"{stem}.wav"
                normalize_wav(raw_wav, final_wav)
                raw_wav.unlink(missing_ok=True)

                # 4) Log into CSV
                ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                writer.writerow([img_path.name, caption, final_wav.name, ts])

            except Exception as e:
                print(f"[WARN] Failed on {img_path.name}: {e}")

    finally:
        f.close()

    print(f"[DONE] Captions: {CAPTION_CSV}")
    print(f"[DONE] Audio files: {AUDIO_DIR}")

if __name__ == "__main__":
    main()
