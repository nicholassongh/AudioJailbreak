import os
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# All three paths below need to be updated
input_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/combined_output.jsonl'
output_file = '/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/text_GPT4o_response_jsonl'
output_jsonl = os.path.join(output_file, 'combined_output.jsonl')

os.makedirs(output_file, exist_ok=True)

with open(input_file, 'r') as f:
    lines = f.readlines()
    updated_lines = []
    i = 1
    for line in lines:
        data = json.loads(line)
        prompt = data.get('prompt', '')

        # Call GPT-4o model
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Use gpt-4o model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        response_text = completion.choices[0].message.content

        print(f"Processing {i} of {len(lines)}")
        print(response_text)
        i += 1

        # Update data and append to output list
        data['response'] = response_text
        updated_lines.append(json.dumps(data) + '\n')

# Write to output file
with open(output_jsonl, 'w') as f:
    f.writelines(updated_lines)
