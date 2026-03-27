import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# TODO: investigate why this runs so slowly

# Set model path
MODEL_CACHE_DIR = "/mnt/data/huggingface/transformers/models"
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"  # Use the text-modality model

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_CACHE_DIR,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_CACHE_DIR,
    device_map="auto",
    trust_remote_code=True
)

def infer_text(prompt: str) -> str:
    """
    Run inference on a single text prompt.

    Args:
        prompt (str): Text prompt

    Returns:
        str: Model's text response
    """
    # Build conversation format
    conversation = [
        {"role": "user", "content": prompt}
    ]

    # Apply conversation template
    text = tokenizer.apply_chat_template(conversation, tokenize=False)

    # Encode input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=256,
        )

    # Decode response
    response = tokenizer.batch_decode(generate_ids[:, inputs['input_ids'].size(1):],
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]

    return response

def process_jsonl(input_file: str, output_file: str):
    """
    Process all text entries in a JSONL file and save results.

    Args:
        input_file (str): Input JSONL file path
        output_file (str): Output JSONL file path
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read and process data
    with open(input_file, 'r') as f:
        lines = f.readlines()
        updated_lines = []
        i = 1
        for line in lines:
            data = json.loads(line)
            prompt = data.get('prompt', '')  # Get text prompt

            # Run model inference
            response_text = infer_text(prompt)

            print(f"Processing {i} of {len(lines)}")
            print(response_text)
            i += 1

            # Update data and append to output list
            data['response'] = response_text
            updated_lines.append(json.dumps(data) + '\n')

    # Write to output file
            with open(output_file, 'w') as f:
                f.writelines(updated_lines)

if __name__ == "__main__":
    # Set input and output file paths
    input_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/combined_output.jsonl'
    output_dir = '/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/text_Qwen2_response_jsonl'
    output_jsonl = os.path.join(output_dir, 'combined_output.jsonl')

    # Process file
    process_jsonl(input_file, output_jsonl)
