#!/usr/bin/env python3
"""
auto-classify.py

Reads segments.csv and audio_features.csv, merges them,
and queries the GPT API for empathy ratings under three conditions:
1) Transcript only
2) Audio features only
3) Combined (transcript + audio features)

Output: classification_results.csv
"""

import csv
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    segments_file = "segments.csv"
    features_file = "audio_features.csv"
    output_file = "classification_results.csv"
    
    # Read features into a dict keyed by (videoId, start, end)
    features_data = {}
    with open(features_file, "r", encoding="utf-8") as ff:
        reader = csv.DictReader(ff)
        for row in reader:
            key = (row["videoId"], row["start"], row["end"])
            features_data[key] = row
    
    # Prepare output
    with open(segments_file, "r", encoding="utf-8") as seg_csv, \
         open(output_file, "w", newline="", encoding="utf-8") as out_csv:
        
        seg_reader = csv.DictReader(seg_csv)
        fieldnames = seg_reader.fieldnames + [
            "transcript_empathy_rating",
            "audio_features_empathy_rating",
            "combined_empathy_rating"
        ]
        
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        writer.writeheader()
        
        for seg_row in seg_reader:
            video_id = seg_row["videoId"]
            start = seg_row["start (M:S)"]
            end = seg_row["end (M:S)"]
            transcript = seg_row.get("transcript", "")
            
            # Look up features
            key = (video_id, start, end)
            audio_feats = features_data.get(key, {})
            
            # Construct prompts
            transcript_prompt = (
                "You are an assistant that rates the level of empathy in a text.\n"
                "Please provide a single integer from 1 to 5, where 1 = low empathy and 5 = high empathy.\n\n"
                f"Transcript:\n{transcript}\n\n"
                "Empathy rating:"
            )
            
            audio_prompt = (
                "You are an assistant that rates the level of empathy in a speaker's audio features.\n"
                "Please provide a single integer from 1 to 5, where 1 = low empathy and 5 = high empathy.\n\n"
                f"Audio features:\n{audio_feats}\n\n"
                "Empathy rating:"
            )
            
            combined_prompt = (
                "You are an assistant that rates the level of empathy using both transcript and audio features.\n"
                "Please provide a single integer from 1 to 5, where 1 = low empathy and 5 = high empathy.\n\n"
                f"Transcript:\n{transcript}\n\n"
                f"Audio features:\n{audio_feats}\n\n"
                "Empathy rating:"
            )
            
            # GPT classification
            seg_row["transcript_empathy_rating"] = get_gpt_rating(transcript_prompt)
            seg_row["audio_features_empathy_rating"] = get_gpt_rating(audio_prompt)
            seg_row["combined_empathy_rating"] = get_gpt_rating(combined_prompt)
            
            writer.writerow(seg_row)

def get_gpt_rating(prompt: str) -> str:
    """
    Sends the prompt to the OpenAI API and returns the stripped response text.
    Adjust 'max_tokens' or 'model' as needed.
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=20,
            temperature=0.0
        )
        # Return only the text of the first choice, stripped of extra whitespace
        rating = response.choices[0].text.strip()
        return rating
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error"

if __name__ == "__main__":
    main()
