import os
import json
from google import genai

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  # Use environment variable for API key

# All three paths below need to be updated
input_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/combined_output.jsonl'
output_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/text_Gemini2.0_flash_response_jsonl'
output_jsonl = os.path.join(output_file, 'combined_output.jsonl')

os.makedirs(output_file, exist_ok=True)

with open(input_file, 'r') as f:
    lines = f.readlines()
    updated_lines = []
    i = 1
    for line in lines:
        data = json.loads(line)
        prompt = data.get('prompt', '')  # Get text prompt

        # Call Gemini 2.0 Flash model for text processing
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[prompt]
        )

        response_text = response.text

        print(f"Processing {i} of {len(lines)}")
        print(response_text)
        i += 1

        # Update data and append to output list
        data['response'] = response_text
        updated_lines.append(json.dumps(data) + '\n')

# Write to output file
        with open(output_jsonl, 'w') as f:
            f.writelines(updated_lines)
