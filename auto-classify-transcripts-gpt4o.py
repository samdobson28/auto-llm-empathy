#!/usr/bin/env python3
"""
auto-classify.py

Reads segments.csv and uses only the transcript (from segments.csv) to query the GPT API for empathy classification.
We disable the wav and combined sections by commenting them out.
The GPT prompt instructs the assistant to return exactly one of:
  'empathetic', 'anti-empathetic', or 'neutral'.

An extra column "match" is added to the output (output-transcripts-gpt4o.csv) along with the original transcript.
A "hit" means the ground truth equals the GPT output.
A "near-miss" is when one is neutral and the other is empathetic or anti-empathetic.
A "miss" is when one is empathetic and the other is anti-empathetic.
Processed results are immediately appended to output-transcripts-gpt4o.csv so that partial results are saved even if interrupted.
"""

import os
import csv
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env file.
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt: str) -> str:
    """
    Calls the GPT API using the new client interface.
    Returns the classification result (one of: 'empathetic', 'anti-empathetic', or 'neutral').
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Adjust model if necessary.
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert empathy classifier. When classifying, choose exactly one "
                        "of the following: empathetic, anti-empathetic, or neutral."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"Error: {e}"

def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase and removing spaces and hyphens.
    """
    return re.sub(r"[\s\-]", "", text.lower())

def compare_match(ground_truth: str, gpt_output: str) -> str:
    """
    Compare the ground truth and GPT output.
    Returns:
      - "hit" if they match exactly,
      - "near-miss" if one is 'neutral' and the other is either 'empathetic' or 'antiempathetic',
      - "miss" if one is 'empathetic' and the other is 'antiempathetic'.
    """
    norm_gt = normalize_text(ground_truth)
    norm_out = normalize_text(gpt_output)
    
    if norm_gt == norm_out:
        return "hit"
    elif ("neutral" in [norm_gt, norm_out]) and (("empathetic" in [norm_gt, norm_out]) or ("antiempathetic" in [norm_gt, norm_out])):
        return "near-miss"
    elif ((norm_gt == "empathetic" and norm_out == "antiempathetic") or 
          (norm_gt == "antiempathetic" and norm_out == "empathetic")):
        return "miss"
    else:
        return "miss"

def main():
    segments_csv = "segments.csv"
    output_csv = "output-transcripts-gpt4o.csv"
    # segments_folder is no longer used since we are only testing transcript
    # segments_folder = "segments"
    
    # Load keys of already processed segments from output-transcripts-gpt4o.csv (if it exists)
    processed_keys = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", newline="", encoding="utf-8") as out_file:
            reader = csv.DictReader(out_file)
            for row in reader:
                key = (row["videoID"], row["start"], row["end"])
                processed_keys.add(key)
    
    # Define output field names (added transcript column)
    fieldnames = ["videoID", "start", "end", "Category", "transcript",
                  "transcript-llm-output", "match"]
    
    # Open output file for appending processed rows.
    with open(output_csv, "a", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        # If file is empty, write header.
        if out_file.tell() == 0:
            writer.writeheader()
            out_file.flush()
        
        # Open segments.csv and process each row.
        with open(segments_csv, "r", newline="", encoding="utf-8") as seg_file:
            reader = csv.DictReader(seg_file)
            try:
                for row in reader:
                    key = (row["videoID"], row["start"], row["end"])
                    if key in processed_keys:
                        print(f"Skipping already processed segment: {key}")
                        continue
                    
                    video_id = row["videoID"]
                    start_str = row["start"]
                    end_str = row["end"]
                    transcript = row.get("transcript", "")
                    category = row.get("Category", "").strip()
                    
                    # Build prompt for transcript only.
                    transcript_prompt = (
                        "Classify the empathy level based solely on the following transcript. "
                        "Answer with exactly one of: empathetic, anti-empathetic, or neutral.\n\n"
                        f"Transcript:\n{transcript}\n\nAnswer:"
                    )
                    
                    transcript_result = call_llm(transcript_prompt)
                    match = compare_match(category, transcript_result)
                    
                    result_row = {
                        "videoID": video_id,
                        "start": start_str,
                        "end": end_str,
                        "Category": category,
                        "transcript": transcript,
                        "transcript-llm-output": transcript_result,
                        "match": match
                    }
                    
                    writer.writerow(result_row)
                    out_file.flush()  # Ensure the row is written to disk immediately.
                    processed_keys.add(key)
                    print(f"Processed segment for video {video_id} from {start_str} to {end_str} -- Match: {match}")
                    
                    # --- The following code for wav and combined prompts is disabled ---
                    # wav_filename = f"{video_id}_{start_str.replace(':', '-')}_{end_str.replace(':', '-')}.wav"
                    # wav_filepath = os.path.join(segments_folder, wav_filename)
                    # if not os.path.exists(wav_filepath):
                    #     print(f"Wav file {wav_filepath} not found for segment {video_id} {start_str}-{end_str}. Skipping wav and combined.")
                    #     continue
                    #
                    # wav_prompt = (
                    #     "Classify the empathy level based solely on the audio file provided. "
                    #     "Assume you can 'listen' to the file at the path below. "
                    #     "Answer with exactly one of: empathetic, anti-empathetic, or neutral.\n\n"
                    #     f"Audio file path: {wav_filepath}\n\nAnswer:"
                    # )
                    #
                    # combined_prompt = (
                    #     "Classify the empathy level based on both the audio file and the transcript provided. "
                    #     "Answer with exactly one of: empathetic, anti-empathetic, or neutral.\n\n"
                    #     f"Audio file path: {wav_filepath}\n\n"
                    #     f"Transcript:\n{transcript}\n\nAnswer:"
                    # )
                    #
                    # wav_result = call_llm(wav_prompt)
                    # combined_result = call_llm(combined_prompt)
                    
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Exiting processing loop.")
            finally:
                out_file.flush()

if __name__ == "__main__":
    main()
