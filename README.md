# Empathy Classification Pipeline

## Overview

This project processes video segments to classify empathy levels using GPT-4o. It integrates transcript and audio feature analysis to determine how empathetic a speaker is.

## Pipeline Structure

1. **`extractor.py`**: Downloads YouTube audio, extracts transcripts, and computes audio features.
2. **`auto-classify.py`**: Uses GPT-4o to classify empathy based on:
   - Transcript alone
   - Audio features alone
   - Both transcript & audio features
3. **`analyze.py`**: Compares GPT-generated ratings with ground truth labels and evaluates performance.

## File Structure

```
project_directory/
├── extractor.py        # Extracts transcripts & audio features
├── auto-classify.py    # Queries GPT-4o for empathy classification
├── analyze.py          # Computes accuracy & error analysis
├── segments.csv        # Video segment metadata (transcript included)
├── audio_features.csv  # Extracted audio features (Librosa-based)
├── classification_results.csv  # GPT-generated empathy ratings
├── analysis_results.csv  # Final analysis output
└── README.md           # Project documentation
```

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
- Sends three types of prompts to GPT-4o:
  - **Transcript only**
  - **Audio features only**
  - **Combined transcript + audio**
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
```

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

## Possible Improvements

- **Enhance audio feature selection** (e.g., pitch variation, speech rate, prosody).
- **Refine GPT prompts** to better interpret numerical audio features.
- **Experiment with weighting** transcript vs. audio input in combined classification.

## Contact

For questions or contributions, reach out to `sed2191@columbia.edu`.

---

**Author:** Sam Dobson  
**Date:** February 2025
