import os
from pydub import AudioSegment

def convert_mp3_to_wav():
    # Set input and output directories
    input_dir = "./audio/jailbreakbench"
    output_dir = "./audio/wav_jailbreakbench"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all MP3 files
    mp3_files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]

    # Convert each file
    for mp3_file in mp3_files:
        # Build full input/output paths
        mp3_path = os.path.join(input_dir, mp3_file)
        wav_file = mp3_file.rsplit('.', 1)[0] + '.wav'  # Only change the extension
        wav_path = os.path.join(output_dir, wav_file)

        # Convert only if the WAV file doesn't exist
        if not os.path.exists(wav_path):
            print(f"Converting {mp3_file} to WAV...")
            try:
                # Read MP3
                audio = AudioSegment.from_mp3(mp3_path)
                # Set sample rate to 16kHz
                audio = audio.set_frame_rate(16000)
                # Export as WAV
                audio.export(wav_path, format='wav')
                print(f"Successfully converted {mp3_file}")
            except Exception as e:
                print(f"Error converting {mp3_file}: {str(e)}")
        else:
            print(f"Skipping {mp3_file} - WAV file already exists")

if __name__ == "__main__":
    convert_mp3_to_wav()
