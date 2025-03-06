# Empathy Classification Pipeline

## Pipeline Structure

1. **`extractor.py`**: Downloads YouTube audio, extracts transcripts, and computes audio features.
2. **`auto-classify-transcripts-[MODEL].py`**: Uses LLM model to classify empathy based on:
   - Transcript
3. **`auto-classify-wavs-[MODEL].py`**: Uses LLM model to classify empathy based on:
   - wav file

## Directory Structure
```

auto-llm-empathy/
├── README.md
├── auto-classify-transcripts-[MODEL].py
├── auto-classify-wavs-[MODEL].py
├── extractor.py
├── requirements.txt
├── output-transcripts-[MODEL].csv
├── output-wavs-[MODEL].csv
├── segments.csv


````

- **`auto-classify-transcripts-[MODEL].py`**: Classifies empathy based on text transcripts only.
- **`auto-classify-wavs-[MODEL].py`**: Classifies empathy based on direct WAV files (GPT-4o audio).
- **`extractor.py`**: Downloads YouTube audio, extracts transcripts, computes features.
- **`requirements.txt`**: Lists Python dependencies.
- **`output-transcripts-[MODEL].csv`**, **`output-wavs-[MODEL].csv`**: Stores classification results.
- **`segments.csv`**: Ground truth metadata (timestamps, categories).

## How It Works

### 1. Extracting Features (`extractor.py`)

- Downloads YouTube audio.
- extracts segments according to time stamps in segments.csv, puts those segmented wavs into segments/

### 2. Classifying Empathy (`auto-classify-transcripts-[MODEL].py`)

- Reads `segments.csv` (which contains video timestamps and transcripts) and segments/ folder for wav files
- Saves results in `output-transcripts-[MODEL].csv`.

### 3. Classifying Empathy (`auto-classify-wavs-[MODEL].py`)

- Reads segments/ folder for wav files
- Saves results in `output-wavs-[MODEL].csv`.

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
