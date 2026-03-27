import json
import os
from io import BytesIO
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import gc

# Set model path to user directory
MODEL_CACHE_DIR = os.path.expanduser("~/qian_jiang/models/Qwen2-Audio-7B-Instruct")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Load model
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    cache_dir=MODEL_CACHE_DIR,
)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    cache_dir=MODEL_CACHE_DIR,
    device_map="auto",
    torch_dtype=torch.float16  # Use half precision
)

def infer_audio(audio_path: str) -> str:
    """
    Run inference on a single audio file.

    Args:
        audio_path (str): Path to the audio file

    Returns:
        str: Model's text response
    """

    conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": audio_path},
    ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        ele['audio_url'],
                        sr=processor.feature_extractor.sampling_rate
                    )[0])

    # Process inputs
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    # Move all inputs to GPU
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate response
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=256,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)
    return response


def process_jsonl(jsonl_path: str, output_dir: str, start_index: int = 0):
    """
    Process all audio files in a JSONL file and save results.

    Args:
        jsonl_path (str): Input JSONL file path
        output_dir (str): Output directory
        start_index (int): JSONL line index to start processing from (default 0)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Copy input file to output directory
    output_jsonl = os.path.join(output_dir, 'sorted_combined_output.jsonl')
    if not os.path.exists(output_jsonl):
        import shutil
        shutil.copy2(jsonl_path, output_jsonl)

    # Read and process data
    data = []
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Create progress tracking file
    progress_file = os.path.join(output_dir, "progress.txt")

    # Resume from last interrupted position if progress file exists and no start index specified
    if os.path.exists(progress_file) and start_index == 0:
        with open(progress_file, 'r') as f:
            last_index = int(f.read().strip())
            print(f"Resuming from last interrupted position: line {last_index}")
            start_index = last_index
    else:
        print(f"Starting from specified position: line {start_index}")

    # Process each sample
    for i, item in enumerate(data[start_index:], start=start_index):
        speech_path = item['speech_path']
        print(f"Processing: {speech_path} (line {i})")

        # Record current line number
        with open(progress_file, 'w') as f:
            f.write(str(i))

        # Use the encapsulated inference function
        response = infer_audio(speech_path)
        item['response'] = response
        print(f"Response generated for {speech_path}")

        # Save results every 10 samples
        if i % 10 == 0 and i > 0:
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Clear GPU memory every N samples
        if i % 20 == 0 and i > 0:
            torch.cuda.empty_cache()
            gc.collect()

def process_batch(batch_items, batch_size=4):
    batch_audios = []
    batch_texts = []

    for item in batch_items:
        conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": item['speech_path']}]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio = librosa.load(item['speech_path'], sr=processor.feature_extractor.sampling_rate)[0]

        batch_audios.append(audio)
        batch_texts.append(text)

    # Batch process inputs
    inputs = processor(text=batch_texts, audios=batch_audios, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Batch generate responses
    generate_ids = model.generate(**inputs, max_new_tokens=256)
    generate_ids = generate_ids[:, inputs['input_ids'].size(1):]

    responses = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return responses

if __name__ == "__main__":
    # Example 1: Process a single audio file
    # audio_path = "path/to/your/audio.wav"
    # response = infer_audio(audio_path)
    # print(f"Response: {response}")

    # Example 2: Process a JSONL file (original functionality)
    jsonl_path = "./convert/sorted_combined_output.jsonl"
    output_dir = "./inference/qwen2_response_jsonl"
    start_index = 1  # Line index to start from, modify as needed
    process_jsonl(jsonl_path, output_dir, start_index)
