import base64
import os
import json
import shutil
from openai import OpenAI
from dotenv import load_dotenv
import datetime

def init_openai_client():
    """Initialize the OpenAI client."""
    #load_dotenv()
    load_dotenv("/home/xiuying.chen/qian_jiang/AudioJailbreak/.env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: Could not get OPENAI_API_KEY from .env file, trying system environment variables")

    if api_key:
        print(f"API key loaded, first 30 characters: {api_key[:30]}...")
        return OpenAI(api_key=api_key)
    else:
        print("Error: Could not get OPENAI_API_KEY, please check environment variables or .env file")
        exit(1)

def infer_audio(audio_path: str) -> str:
    """
    Run inference on an audio file using GPT-4.

    Args:
        audio_path: Path to the audio file

    Returns:
        str: The model's text response
    """
    # Initialize client
    client = init_openai_client()

    # Read audio file
    with open(audio_path, "rb") as audio_file:
        wav_data = audio_file.read()
    encoded_string = base64.b64encode(wav_data).decode('utf-8')

    # Call GPT-4o API
    completion = client.chat.completions.create(
        model="gpt-4o-mini-audio-preview",
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string,
                            "format": "mp3"
                        }
                    }
                ]
            }
        ]
    )

    return completion.choices[0].message.content

def save_response_data(response, output_dir, input_audio_path):
    """
    Save GPT-4 response audio and text data.

    Args:
        response: Response object returned by GPT-4
        output_dir: Output directory path
        input_audio_path: Input audio file path
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(input_audio_path))[0]

        if response.audio and response.audio.transcript:
            return None, None, response.audio.transcript
        else:
            print("No text data in response")
            return None, None, None

    except Exception as e:
        print(f"Error processing response data: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    # Set input and output paths
    input_jsonl = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/combined_output.jsonl"
    output_dir = "/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/gpt4o_response_jsonl"
    output_jsonl = os.path.join(output_dir, "combined_output.jsonl")
    audio_output_dir = os.path.join(output_dir, "response_audio")

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)

    # Copy input file to output directory
    if not os.path.exists(output_jsonl):
        shutil.copy2(input_jsonl, output_jsonl)
        print(f"Copied {input_jsonl} to {output_jsonl}")

    # Read JSONL file
    data = []
    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Limit number of files to process
    max_files = 1000
    data = data[:max_files]

    # Specify starting line
    start_index = 610

    # Process each sample
    for i, item in enumerate(data[start_index:], start=start_index):
        speech_path = item.get('speech_path')
        print(f"Processing file {i+1}/{len(data)}: {speech_path}")

        try:
            # Call inference function
            transcript = infer_audio(speech_path)

            # Update response in JSONL
            print(transcript)
            item['response'] = transcript

            # Save after processing each file
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')

            print(f"File {speech_path} processed successfully!")

        except Exception as e:
            print(f"Error processing file {speech_path}: {str(e)}")
            item['response'] = f"Error: {str(e)}"

            # Save current progress
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print("All files processed!")
