import os
import csv
from pydub import AudioSegment

def m_s_to_millis(time_str):
    """Convert a M:S timestamp to milliseconds."""
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    return (minutes * 60 + seconds) * 1000

def splice_audio(segment, downloads_folder, segments_folder):
    videoID = segment['videoID']
    start_str = segment['start']
    end_str = segment['end']
    
    # Assume full wav file is named using the videoID (e.g. "videoID.wav")
    wav_filepath = os.path.join(downloads_folder, f"{videoID}.wav")
    # Create a filename for the spliced segment; replace ':' with '-' in timestamps
    output_filename = f"{videoID}_{start_str.replace(':', '-')}_{end_str.replace(':', '-')}.wav"
    output_filepath = os.path.join(segments_folder, output_filename)
    
    # Check if the segment already exists; if so, skip splicing
    if os.path.exists(output_filepath):
        print(f"Segment {output_filepath} already exists. Skipping splicing.")
        return output_filepath

    # Ensure the full wav file exists
    if not os.path.exists(wav_filepath):
        print(f"Full wav file {wav_filepath} not found for videoID {videoID}.")
        return None

    # Load the full wav file and splice using timestamps (converted to milliseconds)
    audio = AudioSegment.from_wav(wav_filepath)
    start_ms = m_s_to_millis(start_str)
    end_ms = m_s_to_millis(end_str)
    segment_audio = audio[start_ms:end_ms]
    
    # Export the spliced segment
    segment_audio.export(output_filepath, format="wav")
    print(f"Exported segment to {output_filepath}")
    return output_filepath

def main():
    downloads_folder = "downloads"
    segments_folder = "segments"
    os.makedirs(segments_folder, exist_ok=True)

    segments_csv = "segments.csv"
    with open(segments_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            splice_audio(row, downloads_folder, segments_folder)

if __name__ == "__main__":
    main()
