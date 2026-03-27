import json
import os
from transformers import AutoModel
import librosa
import shutil
import torch
import gc
import signal
from contextlib import contextmanager
import time
import argparse

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Load model globally — only executed once
print("Loading DiVA model...")
load_start_time = time.time()
model = AutoModel.from_pretrained(
    "WillHeld/DiVA-llama-3-v0-8b",
    trust_remote_code=True
)
print(f"DiVA model loaded, elapsed: {time.time() - load_start_time:.2f}s")

def infer_audio(speech_path):
    """
    Run inference on an audio file.

    Args:
        speech_path: Path to the audio file

    Returns:
        str: The model's response
    """
    # Note: the model is now loaded globally and does not need to be loaded inside the function
    try:
        # Load audio
        speech_data, _ = librosa.load(speech_path, sr=16_000)

        # Generate response with a 60-second timeout
        try:
            with time_limit(60):
                with torch.no_grad():
                    response = model.generate([speech_data])[0]

            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            return response

        except TimeoutException:
            print(f"Timeout processing {speech_path}")
            return "Error: Processing timeout"

    except Exception as e:
        print(f"Error processing {speech_path}: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='DiVA model inference')
    parser.add_argument('--start_index', type=int, default=0, help='JSONL line index to start processing from')
    args = parser.parse_args()

    # Create output directory
    output_dir = "./inference/Diva_response_jsonl"
    os.makedirs(output_dir, exist_ok=True)

    # Copy input file to output directory
    input_jsonl = "./convert/wav_combined_output.jsonl"
    output_jsonl = os.path.join(output_dir, "wav_combined_output.jsonl")
    shutil.copy2(input_jsonl, output_jsonl)

    # Read and process data
    data = []
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Create progress tracking file
    progress_file = os.path.join(output_dir, "progress.txt")

    # Resume from last interrupted position if progress file exists and no start index specified
    if os.path.exists(progress_file) and args.start_index == 0:
        with open(progress_file, 'r') as f:
            last_index = int(f.read().strip())
            print(f"Resuming from last interrupted position: line {last_index}")
            start_index = last_index
    else:
        start_index = args.start_index
        print(f"Starting from specified position: line {start_index}")

    # Process each sample
    for i, item in enumerate(data[start_index:], start=start_index):
        speech_path = item['speech_path']
        print(f"Processing: {speech_path} (line {i})")

        # Record current line number
        with open(progress_file, 'w') as f:
            f.write(str(i))

        # Call inference function
        response = infer_audio(speech_path)

        # Update response field
        print(response)
        item['response'] = response
        print(f"Response generated for {speech_path}")

        # Save after processing each file
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

        # Log error line numbers
        if response.startswith("Error"):
            with open(os.path.join(output_dir, "error_log.txt"), 'a') as f:
                f.write(f"Error at line {i}: {response}\n")

    # Update progress file after completion
    with open(progress_file, 'w') as f:
        f.write(str(len(data)))

    print("Processing completed!")
