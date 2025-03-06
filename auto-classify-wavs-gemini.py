import os
import sys
import csv
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

#########################
# Configuration
#########################

load_dotenv()
# Use your Gemini API key from Google AI Studio
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Initialize Gemini client (using the Gemini Developer API with v1alpha endpoints)
client = genai.Client(api_key=google_api_key, http_options={'api_version': 'v1alpha'})
# Set your chosen Gemini model (e.g. Gemini 2.0 Flash)
MODEL = "gemini-2.0-flash"

SEGMENTS_DIR = "segments/"
SEGMENTS_CSV = "segments.csv"
OUTPUT_CSV = "output-wavs-gemini.csv"

PROMPT = (
    "Please classify this speech audio as exactly one of the following words "
    "only: empathetic, neutral, or anti-empathetic. Do not provide any additional text."
)

#########################
# Load Ground Truth
#########################

ground_truth = {}
if os.path.exists(SEGMENTS_CSV):
    with open(SEGMENTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["videoID"], row["start"], row["end"])
            ground_truth[key] = row["Category"]

#########################
# Helper Functions
#########################

def encode_audio(file_path: str) -> bytes:
    """Read WAV file and return its bytes."""
    with open(file_path, "rb") as audio_file:
        return audio_file.read()

def extract_video_info(filename: str):
    """
    Extract (videoID, start, end) from filename e.g. 'videoID_0-20_0-33.wav'.
    Returns (videoID, start, end).
    """
    match = re.match(r"([-a-zA-Z0-9_]+)_(\d+)-(\d+)_(\d+)-(\d+)\.wav", filename)
    if match:
        videoID = match.group(1)
        start = f"{int(match.group(2))}:{int(match.group(3)):02d}"
        end = f"{int(match.group(4))}:{int(match.group(5)):02d}"
        return (videoID, start, end)
    return (filename, "", "")

def compare_classification(pred: str, actual: str):
    """
    Compare predicted vs actual category.
      - 'hit' if they match exactly,
      - 'near-miss' if one is "neutral" and the other is either "empathetic" or "anti-empathetic",
      - otherwise, 'miss'.
    """
    pred_lower = pred.lower().strip()
    actual_lower = actual.lower().strip()
    if pred_lower == actual_lower:
        return "hit"
    elif ((pred_lower == "neutral" and actual_lower in ["empathetic", "anti-empathetic"]) or
          (actual_lower == "neutral" and pred_lower in ["empathetic", "anti-empathetic"])):
        return "near-miss"
    else:
        return "miss"

def classify_audio(file_path: str):
    """
    Reads the audio file, creates an audio part, and sends both the audio and a text prompt
    to Gemini. Parses the text response as the classification.
    """
    audio_bytes = encode_audio(file_path)
    # Create an audio part from the raw bytes; ensure the MIME type matches your file (e.g., "audio/wav")
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
    
    # Prepare contents: first the audio part, then the text prompt.
    contents = [audio_part, PROMPT]
    
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig()
    )
    
    classification = response.text.strip().lower() if response.text else "NO_RESPONSE"
    # Post-process the text response to extract one of the valid labels
    valid = {"empathetic", "neutral", "anti-empathetic"}
    if classification not in valid:
        found_label = None
        for label in valid:
            if label in classification:
                found_label = label
                break
        if found_label:
            classification = found_label
        else:
            classification = "NO_RESPONSE"
    
    return classification

#########################
# Main
#########################

def main():
    # Ensure CSV has headers
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["videoID", "start", "end", "Category", "classification-llm-output", "match"])
    
    # Build a set of already processed segments
    already_processed = set()
    with open(OUTPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            already_processed.add((row["videoID"], row["start"], row["end"]))
    
    files = sorted([f for f in os.listdir(SEGMENTS_DIR) if f.endswith(".wav")])
    for filename in files:
        videoID, start, end = extract_video_info(filename)
        if (videoID, start, end) in already_processed:
            print(f"Skipping already-processed segment: {filename}")
            continue

        print(f"Processing: {filename}")
        file_path = os.path.join(SEGMENTS_DIR, filename)
        try:
            classification = classify_audio(file_path)
            cat = ground_truth.get((videoID, start, end), "")
            match = compare_classification(classification, cat)
            
            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([videoID, start, end, cat, classification, match])
                f.flush()
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Classification complete! Results in {OUTPUT_CSV}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial results saved to {OUTPUT_CSV}")
        sys.exit(0)
