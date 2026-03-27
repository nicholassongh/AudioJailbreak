import json
import librosa
import os

def get_audio_duration(audio_path):
    """Get the duration of an audio file."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Warning: Could not get duration for {audio_path}: {e}")
        return float('inf')  # Return infinity so errored files sort to the end

# Read the original JSONL file
input_path = "./convert/combined_output.jsonl"
output_path = "./convert/sorted_combined_output.jsonl"

# Read all data
data = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Get the duration of each audio file and build sort info
audio_info = []
for item in data:
    # Check if the audio path is empty
    if not item.get('speech_path'):
        print(f"Skipping item with empty audio path: {item}")
        continue

    duration = get_audio_duration(item['speech_path'])
    audio_info.append({
        'original_index': item['index'],
        'duration': duration,
        'data': item
    })

# Sort by duration
audio_info.sort(key=lambda x: x['duration'])

# Take only the first 500 and re-index before saving
with open(output_path, 'w', encoding='utf-8') as f:
    for new_index, info in enumerate(audio_info[:500]):
        # Create a new data item, keeping original content, only updating the index
        output_item = info['data'].copy()
        output_item['index'] = new_index

        # Write to JSONL file
        json.dump(output_item, f, ensure_ascii=False)
        f.write('\n')

print(f"Sorting complete, output saved to: {output_path}")
print(f"Processed and output the first 500 audio files")
