import base64
import json
import os
import shutil
from openai import AzureOpenAI
from dotenv import load_dotenv

ENDPOINT = "https://hamming.cognitiveservices.azure.com/"
MODEL_NAME = "gpt-4o"
DEPLOYMENT = "gpt-4o"
API_VERSION = "2024-12-01-preview"


def init_client() -> AzureOpenAI:
    """Initialize the Azure OpenAI client using credentials from the .env file."""
    load_dotenv()
    api_key = os.getenv("AZURE_API_KEY")
    if not api_key:
        raise ValueError("AZURE_API_KEY not found — add it to your .env file")
    return AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=api_key,
        api_version=API_VERSION,
    )


def infer_audio(audio_path: str) -> str:
    """
    Run inference on an audio file using Azure-hosted GPT-4.1.

    Args:
        audio_path: Path to the audio file (mp3 or wav)

    Returns:
        str: The model's text response
    """
    client = init_client()

    with open(audio_path, "rb") as f:
        audio_data = f.read()
    encoded = base64.b64encode(audio_data).decode("utf-8")

    ext = os.path.splitext(audio_path)[-1].lstrip(".").lower()
    audio_format = "wav" if ext == "wav" else "mp3"

    completion = client.chat.completions.create(
        model=DEPLOYMENT,
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded,
                            "format": audio_format,
                        },
                    }
                ],
            }
        ],
    )

    return completion.choices[0].message.content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Azure GPT-4.1 audio inference")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--start_index", type=int, default=0, help="JSONL line index to start from")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Copy input to output if output doesn't exist yet
    if not os.path.exists(args.output):
        shutil.copy2(args.input, args.output)

    data = []
    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    progress_file = args.output + ".progress"
    start_index = args.start_index
    if start_index == 0 and os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            start_index = int(f.read().strip())
        print(f"Resuming from line {start_index}")

    for i, item in enumerate(data[start_index:], start=start_index):
        speech_path = item.get("speech_path", "")
        print(f"Processing {i + 1}/{len(data)}: {speech_path}")

        try:
            item["response"] = infer_audio(speech_path)
            print(item["response"])
        except Exception as e:
            print(f"Error on {speech_path}: {e}")
            item["response"] = f"Error: {e}"

        # Save progress after every sample
        with open(args.output, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        with open(progress_file, "w") as f:
            f.write(str(i + 1))

    # Clean up progress file on successful completion
    if os.path.exists(progress_file):
        os.remove(progress_file)

    print("All samples processed.")
