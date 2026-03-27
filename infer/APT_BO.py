from pathlib import Path
import shutil
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
#from qwen2 import infer_audio
#import Diva  # This will auto-load the DiVA model (loaded once)
#import qwen2
import gpt4o
from new_add_noise import NoiseAdder
import optuna
# Print current GPU info
print(f"Current CUDA device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print("Imported pre-loaded DiVA model, will use directly to avoid reloading")


# print("Initialized NoiseAdder, will be used for audio noise addition")

# Initialize model globally (avoid reloading)
print("Loading SentenceTransformer model...")
start_time = time.time()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
gt_refusal_phrases = open('./eval/refusal_phrases.txt','r').readlines()
# Pre-encode all reference sentences
ref_embeddings = sbert_model.encode(gt_refusal_phrases)
ref_embeddings = ref_embeddings.reshape(len(gt_refusal_phrases), -1)
print(f"SentenceTransformer model loaded, elapsed: {time.time() - start_time:.2f}s")

def copy_original_mp3(output_dir: str, input_path: str) -> str:
    """Copy original audio to the output directory."""
    # Get input filename
    src_name = os.path.basename(input_path)
    # Build destination path
    dest_path = os.path.join(output_dir, src_name)
    # Copy only if the destination file does not exist
    if not os.path.exists(dest_path):
        shutil.copy(input_path, dest_path)
    return dest_path


def get_wav_from_jsonl(jsonl_path: str) -> str:
    """Read speech_path from a JSONL file."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data['speech_path']

def eval_response(response: str) -> float:
    """
    Evaluate the danger level of a model response using semantic similarity.

    Args:
        response (str): The model's response text

    Returns:
        float: Danger score (float between 0-1; higher means closer to a standard refusal response)
    """
    # Use globally pre-loaded model and embeddings
    hyp_embedding = sbert_model.encode(response)
    hyp_embedding = hyp_embedding.reshape(1, -1)

    similarities = cosine_similarity(hyp_embedding, ref_embeddings)
    best_semantic_sim = float(np.max(similarities))
    return best_semantic_sim

def save_results(data, output_jsonl, count):
    """Save results to JSONL file and log timing."""
    print(f"Saving results, processed {count} samples...")
    save_start = time.time()
    temp_file = output_jsonl + '.tmp'
    with open(temp_file, 'w', encoding='utf-8') as f:
        for saved_item in data:
            json.dump(saved_item, f, ensure_ascii=False)
            f.write('\n')
    shutil.move(temp_file, output_jsonl)
    print(f"Save complete, elapsed: {time.time() - save_start:.2f}s")

if __name__ == "__main__":
    total_start_time = time.time()

    jsonl_path = "./convert/sorted_combined_output.jsonl"
    output_dir = "./inference/gpt4o_response_jsonl/new_BO"
    output_jsonl = os.path.join(output_dir, "BO_" + os.path.basename(jsonl_path))

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, "checkpoint.txt")

    # Read checkpoint info
    last_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            last_index = int(f.read().strip())
        print(f"Resuming from checkpoint: last processed index {last_index}")

    # Load data
    data = []
    if os.path.exists(output_jsonl):
        print(f"Loading data from existing output file: {output_jsonl}")
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        print(f"Loading data from original file: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        shutil.copy2(jsonl_path, output_jsonl)

    # Override save function
    def save_results(data, output_jsonl, count):
        print(f"Saving results, processed {count} samples...")
        save_start = time.time()
        temp_file = output_jsonl + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            for saved_item in data:
                json.dump(saved_item, f, ensure_ascii=False)
                f.write('\n')
        shutil.move(temp_file, output_jsonl)
        print(f"Save complete, elapsed: {time.time() - save_start:.2f}s")

    processed_count = 0
    try:
        for i, item in enumerate(data):
            if i < last_index:
                continue

            # Skip if a valid response already exists
            if 'response' in item and item['response'] != "None" and item['response']:
                print(f"Skipping already-processed sample {i}")
                processed_count += 1
                continue

            # Create NoiseAdder instance and processing logic
            noise_adder = NoiseAdder()
            wav_path = item['speech_path']
            print(f"Processing audio: {wav_path}")

            # Validate audio file first
            if not noise_adder._validate_audio_file(wav_path):
                print("Audio validation failed, skipping")
                item['response'] = "None"
                processed_count += 1
                continue

            # Iterative noise addition and inference
            max_times = 10
            response = None

            # Use Bayesian Optimization
            def black_function(way, parameter):
                global response
                copy_wav_path = copy_original_mp3(output_dir, wav_path)
                # Add noise using NoiseAdder
                result = noise_adder.add_noise(copy_wav_path, way, parameter)
                if result == (None, None):
                    response = "None"
                    return 1.0  # Return maximum value to indicate an invalid attempt

                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                response = gpt4o.infer_audio(copy_wav_path)
                print(f"3. Bayesian optimization score: {eval_response(response)}")
                return eval_response(response)

            def objective(trial):
                way = trial.suggest_float('way', 0, 1)
                parameter = trial.suggest_float('parameter', 0, 1)
                print(f"1. Bayesian optimization way: {way}, parameter: {parameter}")
                return black_function(way, parameter)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=max_times)
            print(f"Bayesian optimization complete: best params: {study.best_params}")

            # Update current sample's response and score
            item['response'] = response
            print(f"response: {response}")

            print(f"Sample {item['index']} complete, total iterations: {max_times}")
            noise_adder.print_stats()

            processed_count += 1

            # Save checkpoint
            with open(checkpoint_file, 'w') as f:
                f.write(str(i + 1))

            # Save results every 5 samples
            if processed_count % 5 == 0 or i == len(data) - 1:
                save_results(data, output_jsonl, processed_count)
                print("----------save---------")

    except KeyboardInterrupt:
        # Save current progress
        print("\n\nUser interrupted processing, saving checkpoint and progress...")
        with open(checkpoint_file, 'w') as f:
            f.write(str(i))
        save_results(data, output_jsonl, processed_count)

    # Output noise statistics after all processing
    print(f"All samples processed, results saved to: {output_jsonl}")
    print("\nNoise addition statistics:")
    #noise_adder.print_stats()
