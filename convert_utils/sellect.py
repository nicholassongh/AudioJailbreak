import json
import os
import re

def extract_filename_core(path):
    """Extract the part of a file path between the last / and ."""
    # Extract using regex
    match = re.search(r'/([^/]+)\.[^.]+$', path)
    if match:
        return match.group(1)
    return None

def extract_matching_entries():
    # Define file paths
    combined_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/inference/SALMONN_response_jsonl/combined_output.jsonl"
    sorted_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/sorted_combined_output.jsonl"

    # Determine output file path
    output_dir = os.path.dirname(combined_path)
    output_path = os.path.join(output_dir, "sellect_combined_output.jsonl")

    print(f"Starting processing...")
    print(f"Extracting filename cores from {sorted_path}")
    print(f"Matching records in {combined_path}")
    print(f"Outputting to {output_path}")

    # Extract filename core set from sorted_combined_output.jsonl
    filename_cores = set()
    with open(sorted_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'speech_path' in data:
                    core = extract_filename_core(data['speech_path'])
                    if core:
                        filename_cores.add(core)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line from sorted file: {line[:50]}...")

    print(f"Extracted {len(filename_cores)} unique filename cores")

    # Filter matching entries from combined_output.jsonl
    matched_entries = []
    with open(combined_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if 'speech_path' in data:
                    core = extract_filename_core(data['speech_path'])
                    if core and core in filename_cores:
                        matched_entries.append(data)
                        # Remove matched filename cores from the set to improve subsequent matching efficiency
                        filename_cores.remove(core)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num} from combined file: {line[:50]}...")

    print(f"Found {len(matched_entries)} matching entries")

    # Check for unmatched filename cores
    if filename_cores:
        print(f"Warning: {len(filename_cores)} filename cores had no matching entries")

    # Write matching entries to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in matched_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Processing complete. Matching entries written to {output_path}")

if __name__ == "__main__":
    extract_matching_entries()
