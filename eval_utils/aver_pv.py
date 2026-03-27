#!/usr/bin/env python
import json
import os

def calculate_average_policy_violation(file_path):
    # Read JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract all policy_violation values
    pv_values = [data[category]["policy_violation"] for category in data]

    # Compute average
    average_pv = sum(pv_values) / len(pv_values) if pv_values else 0

    # Add average to the original data
    data["average_policy_violation"] = average_pv

    # Write back to file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    return average_pv

if __name__ == "__main__":
    file_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/eval/text_qwen2/policy_violation_output.jsonl"
    avg = calculate_average_policy_violation(file_path)
    print(f"Average policy violation: {avg}")
