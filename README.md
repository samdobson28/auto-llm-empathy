# Empathy Classification Pipeline

## Pipeline Structure

1. **`extractor.py`**: Downloads YouTube audio, extracts transcripts, and computes audio features.
2. **`auto-classify-transcripts-gpt4o.py`**: Uses GPT-4o to classify empathy based on:
   - Transcript
3. **`auto-classify-wavs-gpt4o.py`**: Uses GPT-4o to classify empathy based on:
   - wav file

## Directory Structure
```

auto-llm-empathy/
├── README.md
├── auto-classify-transcripts-gpt4o.py
├── auto-classify-wavs-gpt4o.py
├── extractor.py
├── requirements.txt
├── output-transcripts-gpt4o.csv
├── output-wavs-gpt4o.csv
├── segments.csv


````

- **`auto-classify-transcripts-gpt4o.py`**: Classifies empathy based on text transcripts only.
- **`auto-classify-wavs-gpt4o.py`**: Classifies empathy based on direct WAV files (GPT-4o audio).
- **`extractor.py`**: Downloads YouTube audio, extracts transcripts, computes features.
- **`requirements.txt`**: Lists Python dependencies.
- **`output-transcripts-gpt4o.csv`**, **`output-wavs-gpt4o.csv`**: Stores classification results.
- **`segments.csv`**: Ground truth metadata (timestamps, categories).

## How It Works

### 1. Extracting Features (`extractor.py`)

- Downloads YouTube audio.
- extracts segments according to time stamps in segments.csv, puts those segmented wavs into segments/

### 2. Classifying Empathy (`auto-classify.py`)

- Reads `segments.csv` (which contains video timestamps and transcripts).
- **Optionally processes WAV files directly** when transcripts are unavailable.
- Sends multiple prompts to GPT-4o:
  - **Transcript only**
- Saves results in `output-transcripts.csv`.

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

#### Extract wavs

```sh
python extractor.py
```

#### Classify Empathy

```sh
python auto-classify.py
```

_(Adjust arguments to switch between transcript-only, audio features, or raw WAV mode.)_

## Contact

For questions or contributions, reach out to `sed2191@columbia.edu`.

---

**Author:** Sam Dobson  
**Date:** February 2025
