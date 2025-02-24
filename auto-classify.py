#!/usr/bin/env python3
"""
auto-classify.py

Reads segments.csv and audio_features.csv, merges them,
and queries the GPT API for empathy ratings under three conditions:
1) Transcript only
2) Audio features only
3) Combined transcript and audio features

If classification_results.csv exists, it skips segments that have already been processed.

Outputs: classification_results.csv
"""

import csv
import os
from dotenv import load_dotenv
import openai

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    segments_file = "segments.csv"
    features_file = "audio_features.csv"
    output_file = "classification_results.csv"
    
    # Load already processed segments (if any) from classification_results.csv
    processed_keys = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as out_csv:
            reader = csv.DictReader(out_csv)
            for row in reader:
                key = (row["videoId"], row["start (M:S)"], row["end (M:S)"])
                processed_keys.add(key)
    
    # Read audio features into a dictionary keyed by (videoId, start (M:S), end (M:S))
    features_data = {}
    with open(features_file, "r", encoding="utf-8") as ff:
        reader = csv.DictReader(ff)
        for row in reader:
            key = (row["videoId"], row["start (M:S)"], row["end (M:S)"])
            features_data[key] = row
    
    # Determine file mode: append if file exists, else write new with header.
    file_exists = os.path.exists(output_file)
    mode = "a" if file_exists else "w"
    
    with open(segments_file, "r", encoding="utf-8") as seg_csv, \
         open(output_file, mode, newline="", encoding="utf-8") as out_csv:
        
        seg_reader = csv.DictReader(seg_csv)
        fieldnames = seg_reader.fieldnames + [
            "transcript_empathy_rating",
            "audio_features_empathy_rating",
            "combined_empathy_rating"
        ]
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for seg_row in seg_reader:
            key = (seg_row["videoId"], seg_row["start (M:S)"], seg_row["end (M:S)"])
            if key in processed_keys:
                print(f"Skipping already processed segment: {key}")
                continue
            
            video_id = seg_row["videoId"]
            start = seg_row["start (M:S)"]
            end = seg_row["end (M:S)"]
            transcript = seg_row.get("transcript", "")
            audio_feats = features_data.get(key, {})
            
            # Build a readable summary of the audio features
            audio_prompt_details = (
                f"Duration (sec): {audio_feats.get('duration_sec', 'N/A')}\n"
                f"Tempo (BPM): {audio_feats.get('tempo_bpm', 'N/A')}\n"
                f"MFCC (mean): {audio_feats.get('mfcc_mean', 'N/A')}\n"
                f"Tonnetz (mean): {audio_feats.get('tonnetz_mean', 'N/A')}\n"
                f"Silence Ratio: {audio_feats.get('silence_ratio', 'N/A')}\n"
            )
            
            transcript_prompt = (
                "You are an assistant that rates the level of empathy in a text.\n"
                "Please provide a single integer from 1 to 5, where 1 = low empathy and 5 = high empathy.\n\n"
                f"Transcript:\n{transcript}\n\n"
                "Empathy rating:"
            )
            
            audio_prompt = (
                "You are an assistant that rates the level of empathy based on a speaker's audio features.\n"
                "The following quantitative features have been extracted:\n\n"
                f"{audio_prompt_details}\n"
                "Please provide a single integer from 1 to 5, where 1 = low empathy and 5 = high empathy.\n"
                "Empathy rating:"
            )
            
            combined_prompt = (
                "You are an assistant that rates the level of empathy using both transcript and audio features.\n\n"
                f"Transcript:\n{transcript}\n\n"
                "Audio features:\n"
                f"{audio_prompt_details}\n"
                "Please provide a single integer from 1 to 5, where 1 = low empathy and 5 = high empathy.\n"
                "Empathy rating:"
            )
            
            seg_row["transcript_empathy_rating"] = get_gpt_rating(transcript_prompt)
            seg_row["audio_features_empathy_rating"] = get_gpt_rating(audio_prompt)
            seg_row["combined_empathy_rating"] = get_gpt_rating(combined_prompt)
            
            writer.writerow(seg_row)
            processed_keys.add(key)
            print(f"Processed segment for video {video_id} from {start} to {end}")

def get_gpt_rating(prompt: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error"


if __name__ == "__main__":
    main()
