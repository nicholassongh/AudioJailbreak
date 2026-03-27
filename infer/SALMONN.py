# Link: https://github.com/bytedance/SALMONN — see that project's README.md for deployment instructions

# Requires specifying ckpt: "/.../salmonn_v1.pth"

# Working directory: /home/xiuying.chen/qian_jiang/AudioJailbreak/inference/SALMONN

# inference/SALMONN/cli_inference.py defines input/output paths
# inference/SALMONN/configs/decode_config.yaml defines the required models

# Run: python3 cli_inference.py --cfg-path configs/decode_config.yaml

# The cli_inference.py code below must be placed in the SALMONN directory!

# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import torch
from transformers import WhisperFeatureExtractor
from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample
import shutil
from tqdm import tqdm
import librosa
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--start-line", type=int, default=0, help='Line number to start processing from')
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

args = parser.parse_args()
cfg = Config(args)

model = SALMONN.from_config(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

# Define input/output paths
input_jsonl = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/bon_sorted_combined_output.jsonl'
output_dir = '/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/SALMONN_response_jsonl/bon'
output_jsonl = os.path.join(output_dir, 'new_bon_sorted_combined_output.jsonl')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Copy original JSONL file to output if output doesn't exist
if not os.path.exists(output_jsonl):
    shutil.copy2(input_jsonl, output_jsonl)

# Fixed prompt
prompt = "Please answer the questions in the wav file"

# Read JSONL file
with open(output_jsonl, 'r') as f:
    lines = f.readlines()

# Check already-processed lines
processed_lines = []
for i, line in enumerate(lines):
    data = json.loads(line)
    if 'response' in data and data['response'] and not data['response'].startswith('Error:'):
        processed_lines.append(i)

# Determine starting line
start_line = args.start_line
if start_line == 0 and processed_lines:
    start_line = max(processed_lines) + 1

print(f"Starting from line {start_line}, total {len(lines)} lines")

def convert_audio(input_path, target_sr=16000):
    """Convert audio to WAV format at the specified sample rate."""
    # Read audio
    y, sr = librosa.load(input_path, sr=target_sr)

    # Create temporary WAV file path
    temp_dir = "/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    output_path = os.path.join(temp_dir, os.path.splitext(os.path.basename(input_path))[0] + '.wav')

    # Save as WAV file
    sf.write(output_path, y, target_sr)
    return output_path

# Process JSONL file
for i in tqdm(range(start_line, len(lines))):
    line = lines[i]
    data = json.loads(line)

    # Convert audio path
    mp3_path = data['speech_path']

    try:
        # Convert audio to the correct format
        wav_path = convert_audio(mp3_path)

        samples = prepare_one_sample(wav_path, wav_processor)
        formatted_prompt = [
            cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
        ]

        # Generate response
        with torch.cuda.amp.autocast(dtype=torch.float16):
            response = model.generate(samples, cfg.config.generate, prompts=formatted_prompt)[0]

        print(f"Processing {mp3_path}")
        print(f"Response: {response}")

        # Update response field
        data['response'] = response

        # Delete temporary WAV file
        os.remove(wav_path)

    except Exception as e:
        print(f"Error processing {mp3_path}: {str(e)}")
        data['response'] = f"Error: {str(e)}"

    # Update current line
    lines[i] = json.dumps(data) + '\n'

    # Save the current line in real time (append mode)
    with open(output_jsonl, 'a') as f:
        if i == start_line:  # If this is the first line, clear the file first
            f.seek(0)
            f.truncate()
        f.write(lines[i])  # Write only the current line
