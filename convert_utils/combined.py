import os
import json
import glob

def combine_jsonl_files():
    """
    Merge all JSONL files except total.jsonl into a new JSONL file,
    ensure the "index" key increments sequentially from 1,
    and clear the "response" value in all records.
    """
    # Get all JSONL files in the current directory
    jsonl_files = glob.glob("*.jsonl")

    # Exclude total.jsonl
    jsonl_files = [f for f in jsonl_files if f != "total.jsonl"]
    jsonl_files = [f for f in jsonl_files if f != "wav_jailbreakbench.jsonl"]
    jsonl_files = [f for f in jsonl_files if f != "combined_output.jsonl"]

    if not jsonl_files:
        print("❌ No JSONL files found")
        return

    print(f"🔍 Found the following JSONL files: {jsonl_files}")

    # Read the contents of all files
    all_records = []
    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        # Set the response value to empty rather than deleting the key
                        if 'response' in record:
                            record['response'] = ""
                        all_records.append(record)
                    except json.JSONDecodeError:
                        print(f"⚠️ Skipping invalid JSON line: {line[:50]}...")
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {str(e)}")

    print(f"✅ Total records read: {len(all_records)}")

    # Reassign index values
    for i, record in enumerate(all_records, 1):
        record["index"] = i

    # Write to new file
    output_file = "combined_output.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"🎉 Merge complete! Written {len(all_records)} records to {output_file}")

if __name__ == "__main__":
    # Change to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    combine_jsonl_files()
