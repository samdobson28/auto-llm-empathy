#!/usr/bin/env python3
"""
extractor.py

Extracts new audio features for each segment listed in segments.csv.
Uses yt_dlp to download the audio if it hasn't been already.
Outputs a new audio_features.csv with the following columns:
[videoId, start (M:S), end (M:S), duration_sec, tempo_bpm, mfcc_mean, tonnetz_mean, silence_ratio]
"""

import csv
import os
import json
import numpy as np
import yt_dlp
from pydub import AudioSegment
import librosa

def download_audio(youtube_url, video_id, output_path="downloads"):
    """
    Download the audio of a YouTube video as a .wav file using yt_dlp.
    Returns the path to the downloaded file.
    """
    os.makedirs(output_path, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = f"{output_path}/{video_id}.wav"
        return filename

def compute_empathy_features(audio_segment: AudioSegment):
    """
    Compute new audio features for empathy analysis:
    - duration_sec: duration of the segment in seconds
    - tempo_bpm: estimated tempo (beats per minute)
    - mfcc_mean: mean MFCC (13 coefficients)
    - tonnetz_mean: mean tonal centroid (6 dimensions)
    - silence_ratio: ratio of frames with low energy (indicating silence)
    """
    # Duration from pydub segment (in seconds)
    duration_sec = len(audio_segment) / 1000.0

    # Convert AudioSegment to a numpy array and get sampling rate.
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sr = audio_segment.frame_rate

    # Normalize samples (pydub uses integer samples, so scale them)
    samples = samples / np.iinfo(audio_segment.array_type).max if hasattr(audio_segment, "array_type") else samples

    # Compute MFCCs and take mean across time for each coefficient (13 coefficients)
    mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13)
    mfcc_mean = mfccs.mean(axis=1).tolist()  # list of 13 values

    # Compute tonal centroid (tonnetz) features on the harmonic component
    harmonic = librosa.effects.harmonic(samples)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    tonnetz_mean = tonnetz.mean(axis=1).tolist()  # list of 6 values

    # Estimate tempo (BPM) using librosa
    tempo_arr = librosa.beat.tempo(y=samples, sr=sr)
    tempo_bpm = float(tempo_arr[0]) if len(tempo_arr) > 0 else 0.0

    # Compute silence ratio using RMS energy (threshold chosen empirically)
    rms = librosa.feature.rms(y=samples)[0]  # shape: (n_frames,)
    silence_threshold = 0.01  # adjust as needed
    silence_frames = np.sum(rms < silence_threshold)
    silence_ratio = float(silence_frames) / float(len(rms)) if len(rms) > 0 else 0.0

    return {
        "duration_sec": round(duration_sec, 2),
        "tempo_bpm": round(tempo_bpm, 2),
        "mfcc_mean": mfcc_mean,         # will be dumped as JSON string
        "tonnetz_mean": tonnetz_mean,     # will be dumped as JSON string
        "silence_ratio": round(silence_ratio, 4)
    }

def convert_time_to_ms(timestr: str) -> int:
    """
    Convert a time string in M:S or H:M:S format to milliseconds.
    E.g. '2:30' -> 150000 ms
    """
    parts = timestr.split(":")
    parts = [int(p) for p in parts]
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid time format: {timestr}")
    
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds * 1000

def main():
    segments_file = "segments.csv"
    output_file = "audio_features.csv"

    # Updated fieldnames reflecting new features
    fieldnames = [
        "videoId",
        "start (M:S)",
        "end (M:S)",
        "duration_sec",
        "tempo_bpm",
        "mfcc_mean",
        "tonnetz_mean",
        "silence_ratio"
    ]

    # Dictionary to cache downloaded audio paths (so we don't re-download)
    downloaded_videos = {}

    with open(segments_file, "r", encoding="utf-8") as seg_csv, \
         open(output_file, "w", newline="", encoding="utf-8") as out_csv:
        
        reader = csv.DictReader(seg_csv)
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            video_id = row["videoId"]
            start = row["start (M:S)"]
            end = row["end (M:S)"]
            start_ms = convert_time_to_ms(start)
            end_ms = convert_time_to_ms(end)
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Check if audio file is already downloaded
            audio_path = f"downloads/{video_id}.wav"
            if not os.path.exists(audio_path):
                try:
                    print(f"Downloading audio for video {video_id}")
                    audio_path = download_audio(youtube_url, video_id)
                    downloaded_videos[video_id] = audio_path
                except Exception as e:
                    print(f"Failed to download audio for {youtube_url}: {e}")
                    continue  # Skip segments for this video
            
            try:
                # Load the full audio file using pydub
                audio = AudioSegment.from_file(audio_path)
                # Extract the segment using start and end times (in ms)
                segment_audio = audio[start_ms:end_ms]
                
                # Compute new empathy-related features on this segment
                feats = compute_empathy_features(segment_audio)
                
                writer.writerow({
                    "videoId": video_id,
                    "start (M:S)": start,
                    "end (M:S)": end,
                    "duration_sec": feats["duration_sec"],
                    "tempo_bpm": feats["tempo_bpm"],
                    "mfcc_mean": json.dumps(feats["mfcc_mean"]),
                    "tonnetz_mean": json.dumps(feats["tonnetz_mean"]),
                    "silence_ratio": feats["silence_ratio"]
                })
                
                print(f"Processed segment for video {video_id} from {start} to {end}")
            
            except Exception as e:
                print(f"Failed to process segment for {youtube_url} from {start} to {end}: {e}")

if __name__ == "__main__":
    main()
