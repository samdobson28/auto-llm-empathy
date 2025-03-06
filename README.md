# Empathy Classification Pipeline

## Overview

This project processes video segments to classify empathy levels using GPT-4o. It integrates transcript **and** audio feature analysis (including raw WAV inputs) to determine how empathetic a speaker is.

## Pipeline Structure

1. **`extractor.py`**: Downloads YouTube audio, extracts transcripts, and computes audio features.
2. **`auto-classify-transcripts.py`**: Uses GPT-4o to classify empathy based on:
   - Transcript
3. **`auto-classify-wavs.py`**: Uses GPT-4o to classify empathy based on:
   - wav file

## Directory Structure
```

auto-llm-empathy/
├── README.md
├── auto-classify-transcripts.py
├── auto-classify-wavs.py
├── extractor.py
├── requirements.txt
├── output-transcripts.csv
├── output-wavs.csv
├── segments.csv


````

- **`analyze.py`**: Analyzes and compares GPT classifications to ground truth.
- **`auto-classify-transcripts.py`**: Classifies empathy based on text transcripts only.
- **`auto-classify-wavs.py`**: Classifies empathy based on direct WAV files (GPT-4o audio).
- **`chart.py`**, **`counter.py`**: Utility scripts for generating charts, counters, etc.
- **`extractor.py`**: Downloads YouTube audio, extracts transcripts, computes features.
- **`requirements.txt`**: Lists Python dependencies.
- **`output-transcripts.csv`**, **`output-wavs.csv`**: Stores classification results.
- **`segments.csv`**: Ground truth metadata (timestamps, categories).
- **`segments/`**: Contains `.wav` files and segment data.
- **`downloads/`**: Holds downloaded audio/video.
- **`old-files/`**: Archived or older scripts.

## How It Works

### 1. Extracting Features (`extractor.py`)

- Downloads YouTube audio.
- Transcribes speech using Whisper.
- Computes audio features such as:
  - **Tempo (BPM)**
  - **MFCC (Mel-Frequency Cepstral Coefficients)**
  - **Tonnetz (Harmonic features)**
  - **Silence Ratio**

### 2. Classifying Empathy (`auto-classify.py`)

- Reads `segments.csv` (which contains video timestamps and transcripts).
- Reads `audio_features.csv` to match audio features with each segment.
- **Optionally processes WAV files directly** when transcripts are unavailable.
- Sends multiple prompts to GPT-4o:
  - **Transcript only**
  - **Audio features only**
  - **Combined transcript + audio**
  - **Raw WAV** (direct audio classification)
- Saves results in `classification_results.csv`.

### 3. Analyzing Performance (`analyze.py`)

- Compares GPT-4o ratings to ground truth empathy labels.
- Computes:
  - **Mean Absolute Error (MAE)**
  - **Accuracy (%)**
  - **Confusion Matrix & Classification Report**
- Outputs results to `analysis_results.csv`.

## Running the Pipeline

### 1. Install Dependencies

Ensure you have all required Python packages:

```sh
pip install -r requirements.txt
````

### 2. Set Up API Key

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_secret_key_here
```

### 3. Run the Scripts

#### Extract Features

```sh
python extractor.py
```

#### Classify Empathy

```sh
python auto-classify.py
```

_(Adjust arguments to switch between transcript-only, audio features, or raw WAV mode.)_

#### Analyze Results

```sh
python analyze.py
```

## Results Summary

| Method         | Mean Absolute Error | Accuracy (%) |
| -------------- | ------------------- | ------------ |
| Transcript     | 1.87                | 37.5%        |
| Audio Features | 2.00                | 0%           |
| Combined       | 1.99                | 17%          |
| **Raw WAV**    | (New!) 2.10         | 15%          |

## Possible Improvements

- **Enhance audio feature selection** (e.g., pitch variation, speech rate, prosody).
- **Refine GPT prompts** to better interpret numerical audio features.
- **Experiment with weighting** transcript vs. audio input in combined classification.

## Contact

For questions or contributions, reach out to `sed2191@columbia.edu`.

---

**Author:** Sam Dobson  
**Date:** February 2025
