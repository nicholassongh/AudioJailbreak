import os
import json
import shutil
import numpy as np
import librosa
import whisper
import re
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load sentence transformer model for semantic similarity computation
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load Whisper model
whisper_model = whisper.load_model("base")

def setup_directories():
    """
    Create necessary directory structure and copy original audio to the experiment directory.
    """
    # Ensure directories exist
    os.makedirs("experiment/origin_mp3", exist_ok=True)
    os.makedirs("experiment/experiment_mp3", exist_ok=True)
    os.makedirs("experiment/jailbreak_results", exist_ok=True)

    # Copy original audio to the experiment directory
    for file_name in os.listdir("experiment/origin_mp3"):
        if file_name.endswith((".mp3", ".wav")):
            source_path = os.path.join("experiment/origin_mp3", file_name)
            dest_path = os.path.join("experiment/experiment_mp3", file_name)
            shutil.copy2(source_path, dest_path)

    print(f"✅ Original audio copied to experiment directory")

def extract_number_from_filename(filename):
    """
    Extract a number from a filename.
    E.g., extracts 5 from 'jailbreakbench_prompt_5.mp3'

    Args:
        filename: The filename

    Returns:
        The extracted number, or None if not found
    """
    match = re.search(r'jailbreakbench_prompt_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def load_text_prompts():
    """
    Load text prompts from a JSONL file.

    Returns:
        A dict mapping index to prompt text
    """
    prompts_dict = {}
    try:
        with open("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/jailbreakbench.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Use index as key, prompt as value
                if "index" in data and "prompt" in data:
                    prompts_dict[data["index"]] = data["prompt"]
        print(f"✅ Loaded {len(prompts_dict)} text prompts")
        return prompts_dict
    except Exception as e:
        print(f"❌ Failed to load text prompts: {str(e)}")
        return {}

def add_noise(audio_path, method_type):
    """
    Add noise or adjust parameters of an audio file.

    Args:
        audio_path: Path to the audio file
        method_type: Adjustment method (1-6)
            1 - Increase volume
            2 - Decrease volume
            3 - Increase pitch
            4 - Decrease pitch
            5 - Increase speed
            6 - Decrease speed
    """
    try:
        # Read audio
        x = AudioSegment.from_file(audio_path)
        y, sr = librosa.load(audio_path, mono=True)

        # Adjust audio based on method type
        if method_type == 1:  # Increase volume
            #x = x + 2  # Increase by 2dB
            x = x + 15
            x.export(audio_path, format="mp3")

        elif method_type == 2:  # Decrease volume
            #x = x - 2  # Decrease by 2dB
            x = x - 15
            x.export(audio_path, format="mp3")

        elif method_type == 3:  # Increase pitch
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
            # Convert to AudioSegment and save
            MAX_INT16 = 32767
            y_int16 = (y * MAX_INT16).astype(np.int16)
            to_audio = AudioSegment(
                data=y_int16.tobytes(),
                sample_width=y_int16.dtype.itemsize,
                frame_rate=sr,
                channels=1
            )
            to_audio.export(audio_path, format="mp3")

        elif method_type == 4:  # Decrease pitch
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
            # Convert to AudioSegment and save
            MAX_INT16 = 32767
            y_int16 = (y * MAX_INT16).astype(np.int16)
            to_audio = AudioSegment(
                data=y_int16.tobytes(),
                sample_width=y_int16.dtype.itemsize,
                frame_rate=sr,
                channels=1
            )
            to_audio.export(audio_path, format="mp3")

        elif method_type == 5:  # Increase speed
            y = librosa.effects.time_stretch(y, rate=1.1)
            # Convert to AudioSegment and save
            MAX_INT16 = 32767
            y_int16 = (y * MAX_INT16).astype(np.int16)
            to_audio = AudioSegment(
                data=y_int16.tobytes(),
                sample_width=y_int16.dtype.itemsize,
                frame_rate=sr,
                channels=1
            )
            to_audio.export(audio_path, format="mp3")

        elif method_type == 6:  # Decrease speed
            y = librosa.effects.time_stretch(y, rate=0.9)
            # Convert to AudioSegment and save
            MAX_INT16 = 32767
            y_int16 = (y * MAX_INT16).astype(np.int16)
            to_audio = AudioSegment(
                data=y_int16.tobytes(),
                sample_width=y_int16.dtype.itemsize,
                frame_rate=sr,
                channels=1
            )
            to_audio.export(audio_path, format="mp3")

        return True
    except Exception as e:
        print(f"❌ Error processing audio {audio_path}: {str(e)}")
        return False

def analyze_audio(audio_path):
    """
    Analyze audio properties.

    Args:
        audio_path: Path to the audio file

    Returns:
        A dict containing audio properties
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path)

        # 1. Volume analysis
        rms = librosa.feature.rms(y=y)[0]
        db = librosa.amplitude_to_db(rms)

        # 2. Pitch analysis
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        valid_f0 = f0[~np.isnan(f0)]

        # 3. Tempo/speed analysis
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # Return analysis results
        return {
            "volume": {
                "rms_mean": float(np.mean(rms)),
                "db_mean": float(np.mean(db)),
            },
            "pitch": {
                "mean_fundamental_freq": float(np.nanmean(f0)) if len(valid_f0) > 0 else 0,
                "pitch_variation_rate": float(np.mean(np.abs(np.diff(valid_f0)))) if len(valid_f0) > 1 else 0
            },
            "speed": {
                "estimated_tempo_bpm": float(np.asarray(tempo).item()),  # Avoid NumPy warning
                "spectral_change_rate": float(np.mean(np.abs(np.diff(np.abs(librosa.stft(y)), axis=1))))
            }
        }
    except Exception as e:
        print(f"❌ Error analyzing audio {audio_path}: {str(e)}")
        return {
            "volume": {"rms_mean": 0, "db_mean": 0},
            "pitch": {"mean_fundamental_freq": 0, "pitch_variation_rate": 0},
            "speed": {"estimated_tempo_bpm": 0, "spectral_change_rate": 0}
        }

def transcribe_audio(audio_path):
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_path: Path to the audio file

    Returns:
        The transcribed text
    """
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"❌ Error transcribing audio {audio_path}: {str(e)}")
        return ""

def calculate_similarity(text1, text2):
    """
    Calculate the similarity between two pieces of text.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0-1)
    """
    # Similarity is 0 if either text is empty
    if not text1 or not text2:
        return 0.0

    # 1. Compute semantic similarity using the sentence transformer model
    try:
        embedding1 = sentence_model.encode([text1])[0]
        embedding2 = sentence_model.encode([text2])[0]
        semantic_sim = cosine_similarity([embedding1], [embedding2])[0][0]
    except Exception as e:
        print(f"❌ Error computing semantic similarity: {str(e)}")
        semantic_sim = 0

    # 2. Compute word-level similarity using BLEU score
    try:
        reference = [word_tokenize(text1.lower())]
        candidate = word_tokenize(text2.lower())
        # Use smoothing function to avoid warnings
        from nltk.translate.bleu_score import SmoothingFunction
        smooth = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smooth)
    except Exception as e:
        print(f"❌ Error computing BLEU score: {str(e)}")
        bleu_score = 0

    # Combine both similarity scores
    similarity = (semantic_sim * 0.7) + (bleu_score * 0.3)
    return float(similarity)

def restore_audio():
    """
    Restore all audio files to their original state.
    """
    # Delete all files in the experiment directory
    for file_name in os.listdir("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/experiment_mp3"):
        file_path = os.path.join("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/experiment_mp3", file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Copy files from the original directory to the experiment directory
    for file_name in os.listdir("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/origin_mp3"):
        if file_name.endswith((".mp3", ".wav")):
            source_path = os.path.join("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/origin_mp3", file_name)
            dest_path = os.path.join("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/experiment_mp3", file_name)
            shutil.copy2(source_path, dest_path)

def main():
    """
    Main function, executes the full experiment pipeline.
    """
    # Set up directories and copy audio files
    #setup_directories()
    # Restore audio to original state
    restore_audio()
    # Load text prompts, now a dict keyed by index
    original_prompts_dict = load_text_prompts()
    if not original_prompts_dict:
        print("❌ Failed to load text prompts, exiting")
        return

    # Get all audio files in the experiment directory
    audio_files = [f for f in os.listdir("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/experiment_mp3")
                  if f.endswith((".mp3", ".wav"))]

    if not audio_files:
        print("❌ No audio files in the experiment directory, exiting")
        return

    # User selects adjustment method
    print("🔹 Select the property to adjust:")
    print("   1 - Increase volume")
    print("   2 - Decrease volume")
    print("   3 - Increase pitch")
    print("   4 - Decrease pitch")
    print("   5 - Increase speed")
    print("   6 - Decrease speed")

    try:
        method_type = int(input("Enter the corresponding number (1-6): ").strip())
        if method_type < 1 or method_type > 6:
            raise ValueError("Option must be between 1 and 6")
    except ValueError as e:
        print(f"❌ Input error: {str(e)}")
        return

    # Method name mapping
    method_names = {
        1: "volume_up",
        2: "volume_down",
        3: "pitch_up",
        4: "pitch_down",
        5: "speed_up",
        6: "speed_down"
    }

    method_name = method_names[method_type]
    print(f"✅ You selected: {method_name}")

    # Prepare result storage
    all_results = []

    # Execute experiment rounds
    for round_idx in range(1, 6):
        print(f"\n🚀 Starting round {round_idx}/6")

        # Store results for this round
        round_results = {
            "round": round_idx,
            "method": method_name,
            "audio_properties": [],
            "similarities": [],
            "original_prompts": [],
            "transcribed_texts": []
        }

        # Process each audio file
        for audio_file in audio_files:
            # Extract number from filename as index
            file_index = extract_number_from_filename(audio_file)
            audio_path = os.path.join("/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/experiment_mp3", audio_file)

            print(f"  🎵 Processing audio: {audio_file} (index: {file_index})")

            # 1. Add noise
            success = add_noise(audio_path, method_type)
            if not success:
                continue

            # 2. Analyze audio properties
            audio_features = analyze_audio(audio_path)

            # 3. Transcribe audio
            transcribed_text = transcribe_audio(audio_path)

            # 4. Get matching original prompt and calculate similarity
            original_prompt = ""
            if file_index is not None and file_index in original_prompts_dict:
                original_prompt = original_prompts_dict[file_index]
                print(f"    ✓ Matched original prompt (index: {file_index})")
            else:
                print(f"    ⚠️ No original prompt found for index {file_index}")

            similarity = calculate_similarity(original_prompt, transcribed_text)

            # Store results
            round_results["audio_properties"].append({
                "filename": audio_file,
                "index": file_index,
                "features": audio_features
            })
            round_results["similarities"].append({
                "filename": audio_file,
                "index": file_index,
                "score": similarity
            })
            round_results["original_prompts"].append({
                "filename": audio_file,
                "index": file_index,
                "text": original_prompt
            })
            round_results["transcribed_texts"].append({
                "filename": audio_file,
                "index": file_index,
                "text": transcribed_text
            })

            print(f"    📊 Similarity: {similarity:.4f}")

        # Calculate averages
        if round_results["similarities"]:
            avg_similarity = sum(item["score"] for item in round_results["similarities"]) / len(round_results["similarities"])
            round_results["avg_similarity"] = avg_similarity
            print(f"  📈 Round average similarity: {avg_similarity:.4f}")

        # Calculate audio property averages
        if round_results["audio_properties"]:
            avg_features = {
                "volume": {"rms_mean": 0, "db_mean": 0},
                "pitch": {"mean_fundamental_freq": 0, "pitch_variation_rate": 0},
                "speed": {"estimated_tempo_bpm": 0, "spectral_change_rate": 0}
            }

            count = len(round_results["audio_properties"])
            for item in round_results["audio_properties"]:
                features = item["features"]
                avg_features["volume"]["rms_mean"] += features["volume"]["rms_mean"] / count
                avg_features["volume"]["db_mean"] += features["volume"]["db_mean"] / count
                avg_features["pitch"]["mean_fundamental_freq"] += features["pitch"]["mean_fundamental_freq"] / count
                avg_features["pitch"]["pitch_variation_rate"] += features["pitch"]["pitch_variation_rate"] / count
                avg_features["speed"]["estimated_tempo_bpm"] += features["speed"]["estimated_tempo_bpm"] / count
                avg_features["speed"]["spectral_change_rate"] += features["speed"]["spectral_change_rate"] / count

            round_results["avg_audio_properties"] = avg_features

        # Append to total results
        all_results.append(round_results)

        # Save current round results
        result_file = f"/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/results/result_{method_name}_round_{round_idx}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(round_results, f, ensure_ascii=False, indent=2)

        print(f"  💾 Round results saved to: {result_file}")

    # Save all results
    final_result_file = f"/home/xiuying.chen/qian_jiang/AudioJailbreak/experiment/results/all_results_{method_name}.json"
    with open(final_result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 All experiments complete! Final results saved to: {final_result_file}")

if __name__ == "__main__":
    main()
