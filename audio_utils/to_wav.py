import os
import json
import shutil
from pydub import AudioSegment
import re

def convert_mp3_to_wav(source_dirs, target_dir):
    """
    Convert MP3 files in the specified directories to WAV format and save to the target directory.

    Args:
        source_dirs: List of source directories containing MP3 files
        target_dir: Target directory to save WAV files
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Track the number of converted files
    converted_count = 0

    # Iterate over each source directory
    for source_dir in source_dirs:
        print(f"Processing directory: {source_dir}")

        # Iterate over all files in the directory
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith('.mp3'):
                    # Build full paths for source and target files
                    mp3_path = os.path.join(root, file)

                    # Extract filename (without extension)
                    filename_without_ext = os.path.splitext(file)[0]

                    # Build target WAV file path
                    wav_filename = f"{filename_without_ext}.wav"
                    wav_path = os.path.join(target_dir, wav_filename)

                    try:
                        # Load MP3 file with pydub and convert to WAV
                        audio = AudioSegment.from_mp3(mp3_path)

                        # Set to 16kHz sample rate and mono (suitable for most speech models)
                        audio = audio.set_frame_rate(16000).set_channels(1)

                        # Export as WAV format
                        audio.export(wav_path, format="wav")

                        converted_count += 1
                        if converted_count % 10 == 0:
                            print(f"Converted {converted_count} files...")

                    except Exception as e:
                        print(f"Error converting {mp3_path}: {str(e)}")

    print(f"Conversion complete! Converted {converted_count} files.")

def update_jsonl_paths(source_jsonl, target_jsonl):
    """
    Copy a JSONL file and update the audio paths within it.

    Args:
        source_jsonl: Source JSONL file path
        target_jsonl: Target JSONL file path
    """
    print(f"Updating JSONL file: {source_jsonl} -> {target_jsonl}")

    # Read source JSONL file
    with open(source_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        data = json.loads(line)

        # Update speech_path field
        if 'speech_path' in data:
            # Use regex to replace the path
            # Replace ./audio/xxx/filename.mp3 with ./audio/total_wav/filename.wav
            old_path = data['speech_path']
            new_path = re.sub(
                r'./audio/[^/]+/([^/]+)\.mp3',
                r'./audio/total_wav/\1.wav',
                old_path
            )
            data['speech_path'] = new_path

        # Append updated data to list
        updated_lines.append(json.dumps(data, ensure_ascii=False) + '\n')

    # Write to target JSONL file
    with open(target_jsonl, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)

    print(f"JSONL file updated successfully, processed {len(updated_lines)} records.")

if __name__ == "__main__":
    # Define source and target directories
    base_dir = "/home/xiuying.chen/qian_jiang/AudioJailbreak"
    source_dirs = [
        f"{base_dir}/audio/Do_Not_Answer",
        f"{base_dir}/audio/jailbreak_llms",
        f"{base_dir}/audio/jailbreakbench"
    ]
    target_dir = f"{base_dir}/audio/total_wav"

    # Define source and target JSONL files
    source_jsonl = f"{base_dir}/convert/combined_output.jsonl"
    target_jsonl = f"{base_dir}/convert/wav_combined_output.jsonl"

    # Perform MP3 to WAV conversion
    convert_mp3_to_wav(source_dirs, target_dir)

    # Update paths in the JSONL file
    update_jsonl_paths(source_jsonl, target_jsonl)

    print("All tasks complete!")
