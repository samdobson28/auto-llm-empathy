import os
import sys
import csv
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

#########################
# Configuration
#########################

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)
MODEL = "gpt-4o-mini-audio-preview"

SEGMENTS_DIR = "segments/"
SEGMENTS_CSV = "segments.csv"
OUTPUT_CSV = "output-wavs.csv"

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

def encode_audio(file_path: str) -> str:
    """Convert WAV file to base64."""
    import base64
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

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
    """Compare to ground truth: 'hit', 'near-miss', or 'miss'."""
    pred_lower = pred.lower()
    actual_lower = actual.lower()
    if pred_lower == actual_lower:
        return "hit"
    elif pred_lower in ["empathetic","neutral","anti-empathetic"] and \
         actual_lower in ["empathetic","neutral","anti-empathetic"]:
        return "near-miss"
    else:
        return "miss"

def classify_audio(file_path: str):
    """Call GPT-4o (audio) and parse classification label."""
    encoded = encode_audio(file_path)
    response = client.chat.completions.create(
        model=MODEL,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": "wav"}
                    }
                ]
            }
        ]
    )

    resp_dict = response.to_dict()

    # content might be None or a list
    classification = None
    message_obj = resp_dict["choices"][0]["message"]
    content_data = message_obj.get("content")  # Possibly None or list

    if content_data and isinstance(content_data, list):
        # Try to find a transcript
        for chunk in content_data:
            if isinstance(chunk, dict) and "transcript" in chunk:
                classification = chunk["transcript"].strip().lower()
                break

    # If still no classification, check .audio.transcript
    if not classification and isinstance(message_obj.get("audio"), dict):
        classification = message_obj["audio"].get("transcript", "").strip().lower()

    # fallback if not found
    if not classification:
        classification = "NO_RESPONSE"

    # If classification has extra text, do a quick substring check
    valid = {"empathetic","neutral","anti-empathetic"}
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
            writer.writerow(["videoID","start","end","Category","classification-llm-output","match"])

    # Build a set of already processed (videoID, start, end)
    already_processed = set()
    with open(OUTPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # we skip them
            already_processed.add((row["videoID"], row["start"], row["end"]))

    files = sorted([f for f in os.listdir(SEGMENTS_DIR) if f.endswith(".wav")])
    for filename in files:
        videoID, start, end = extract_video_info(filename)

        # If this segment is already in output-wavs.csv, skip it
        if (videoID, start, end) in already_processed:
            print(f"Skipping already-processed segment: {filename}")
            continue

        print(f"Processing: {filename}")
        file_path = os.path.join(SEGMENTS_DIR, filename)
        try:
            classification = classify_audio(file_path)
            cat = ground_truth.get((videoID, start, end), "")
            match = compare_classification(classification, cat)

            # Write step-by-step
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
