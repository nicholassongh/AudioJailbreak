import librosa
import numpy as np
from pydub import AudioSegment

# Function to estimate BPM
def estimate_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


# Function to save audio
def save_audio(y, sr, audio_path):
    MAX_INT16 = 32767
    y_int16 = (y * MAX_INT16).astype(np.int16)
    to_audio = AudioSegment(
        data=y_int16.tobytes(),
        sample_width=y_int16.dtype.itemsize,
        frame_rate=sr,
        channels=1
    )
    to_audio.export(audio_path, format="mp3")

# Main pipeline
def process_audio(audio_file, output_audio_file):
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)

    # Test original BPM value
    original_bpm = estimate_bpm(y, sr)
    print(f"Original audio BPM: {original_bpm:.2f}")

    # Stretch duration to double (halve playback speed, rate=0.5)
    y_slow = librosa.effects.time_stretch(y, rate=0.5)

    # Save processed audio
    save_audio(y_slow, sr, output_audio_file)

    # Test processed BPM value
    processed_bpm = estimate_bpm(y_slow, sr)
    print(f"Processed audio BPM: {processed_bpm:.2f}")


# Example usage
audio_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/jailbreakbench_prompt_1.mp3"
output_audio_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/jailbreakbench_prompt_1_slow.mp3"

process_audio(audio_path, output_audio_path)
