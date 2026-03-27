import json
import os

def change_paths(input_file, output_file):
    """
    Read a JSONL file, modify the speech_path field to keep only the filename,
    and add a new path prefix.
    """
    new_prefix = "/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/gpt4o_response_jsonl/new_bon/"

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line.strip())

            if 'speech_path' in data and data['speech_path']:
                # Extract the filename from the original path
                filename = os.path.basename(data['speech_path'])
                # Create new path
                data['speech_path'] = new_prefix + filename

            # Write the modified data
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_jsonl = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/bon_sorted_combined_output.jsonl"
    output_jsonl = input_jsonl  # Output to the same file

    # First create a temporary file
    temp_output = input_jsonl + ".temp"
    change_paths(input_jsonl, temp_output)

    # Replace the original file
    import shutil
    shutil.move(temp_output, output_jsonl)

    print(f"File processing complete, paths updated: {output_jsonl}")
