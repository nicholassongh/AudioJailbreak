import json
import os

# Define input and output file paths
input_jsonl = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/sorted_combined_output.jsonl'
output_jsonl = '/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/wav_sorted_combined_output.jsonl'

# Store converted data
converted_data = []

# Statistics
total_processed = 0
modified_paths = 0

# Read input JSONL file
with open(input_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():  # Skip blank lines
            continue

        total_processed += 1
        entry = json.loads(line)

        # Get current speech_path
        if 'speech_path' in entry and entry['speech_path']:
            current_path = entry['speech_path']

            # Parse path components
            parts = current_path.split('/')

            # Handle filename and extension
            filename = parts[-1]
            if filename.lower().endswith('.mp3'):
                # Replace extension
                filename = filename[:-4] + '.wav'

                # Build new path: prepend "../../" + keep "audio" + replace top-level dir with "total_wav" + filename
                # e.g. "./audio/jailbreak_llms/jailbreak_llms_prompt_1230.mp3"
                #   -> "../../audio/total_wav/jailbreak_llms_prompt_1230.wav"
                new_path = "../../audio/total_wav/" + filename

                # Update entry
                entry['speech_path'] = new_path
                modified_paths += 1

        # Append to result list
        converted_data.append(entry)

# Write to output JSONL file
with open(output_jsonl, 'w', encoding='utf-8') as f:
    for entry in converted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Conversion complete! Total processed: {total_processed} records, modified {modified_paths} paths")
print(f"Output file: {output_jsonl}")
