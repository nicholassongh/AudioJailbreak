import json
import os
from collections import defaultdict

def calculate_average():
    # Define file paths
    input_path = "/home/xiuying.chen/qian_jiang/AudioJailbreak/eval/SALMONN/sellect_sorted_output.jsonl"
    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, "sellect_aver.json")

    print(f"Processing file: {input_path}")

    try:
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    # Count categories
    category_count = len(data)
    print(f"Found {category_count} categories")

    if category_count == 0:
        print("No valid data found, terminating")
        return

    # Accumulators for all metric sums
    metric_sums = defaultdict(float)
    gpt_metric_sums = defaultdict(float)
    metrics_count = defaultdict(int)

    # Iterate over all categories and accumulate metric values
    for category, metrics in data.items():
        # Process top-level metrics
        for metric_name, metric_value in metrics.items():
            if metric_name != "gpt_score" and isinstance(metric_value, (int, float)):
                metric_sums[metric_name] += metric_value
                metrics_count[metric_name] += 1

        # Process metrics within gpt_score
        if "gpt_score" in metrics and isinstance(metrics["gpt_score"], dict):
            for gpt_metric, gpt_value in metrics["gpt_score"].items():
                if isinstance(gpt_value, (int, float)):
                    gpt_metric_sums[gpt_metric] += gpt_value
                    metrics_count[f"gpt_{gpt_metric}"] += 1

    # Calculate averages
    averages = {}

    # Calculate averages for top-level metrics
    for metric_name, total in metric_sums.items():
        count = metrics_count[metric_name]
        if count > 0:
            averages[metric_name] = total / count

    # Calculate averages for gpt_score metrics
    gpt_averages = {}
    for gpt_metric, total in gpt_metric_sums.items():
        count = metrics_count[f"gpt_{gpt_metric}"]
        if count > 0:
            gpt_averages[gpt_metric] = total / count

    # If there are gpt_score metrics, add them to the results
    if gpt_averages:
        averages["gpt_score"] = gpt_averages

    # Output results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(averages, f, indent=4)

    print(f"Processing complete, averages saved to: {output_path}")

    # Print results preview
    print("\nResults preview:")
    print(json.dumps(averages, indent=2))

if __name__ == "__main__":
    calculate_average()
