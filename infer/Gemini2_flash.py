import os
from google import genai
import json
import argparse

def infer_audio(audio_path):
    """
    Run inference on audio using the Gemini 2.0 Flash model.

    Args:
        audio_path: Path to the audio file

    Returns:
        str: Inference result text
    """
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    try:
        myfile = client.files.upload(file=audio_path)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                'Answer the content of the audio',
                myfile,
            ]
        )
        return response.text
    except Exception as e:
        print(f"Error during inference: {e}")
        raise e

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Gemini 2.0 Flash model inference')
    parser.add_argument('--start_index', type=int, default=0, help='JSONL line index to start processing from')
    args = parser.parse_args()

    input_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/combined_output.jsonl'
    output_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/Gemini2.0_flash_response_jsonl'
    output_jsonl = os.path.join(output_file, 'combined_output.jsonl')
    progress_file = os.path.join(output_file, 'progress.txt')

    os.makedirs(output_file, exist_ok=True)

    # Check for progress file; read last processed line count if it exists
    start_index = args.start_index
    if os.path.exists(progress_file) and args.start_index == 0:
        with open(progress_file, 'r') as f:
            try:
                start_index = int(f.read().strip())
                print(f"Resuming from last interrupted position: line {start_index + 1}")
            except ValueError:
                print("Progress file content is invalid, starting from the beginning.")
    else:
        print("Starting from the beginning.")

    with open(input_file, 'r') as f:
        lines = f.readlines()
        total_lines = len(lines)

        for i, line in enumerate(lines):
            if i < start_index:
                continue

            data = json.loads(line)
            audio_path = data['speech_path']

            try:
                response_text = infer_audio(audio_path)
                print(f"Processing {i + 1} of {total_lines}")
                print(response_text)

                data['response'] = response_text
                updated_line = json.dumps(data) + '\n'

                # Write response to file immediately
                with open(output_jsonl, 'a') as outfile:
                    outfile.write(updated_line)

                # Write progress to file
                with open(progress_file, 'w') as pf:
                    pf.write(str(i + 1) + '\n')  # Record number of processed lines

            except Exception as e:
                print(f"Error processing line {i + 1}: {e}")
                print("Saving current progress and exiting.")
                exit()

        print("Processing complete!")
        # Clear progress file
        open(progress_file, 'w').close()
