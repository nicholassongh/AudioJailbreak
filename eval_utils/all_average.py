import json
import os
from collections import defaultdict

# Model and directory configuration
MODELS = ["Diva", "Gemini2.0_flash", "gpt4o", "LLama_Omni", "qwen2", "SALMONN"]
DIRS = ["BO", "bon"]
OUTPUT_PATH = "./eval/average_result.json"
CATEGORY_STATS_PATH = "./convert/category_statistics.txt"

# Metrics to compute
METRICS = ["semantic_sim", "dict_confused", "dict_dsr"]
GPT_METRICS = ["harmfulness", "policy_violation", "relevance"]

def load_category_weights():
    """Load category weight information."""
    weights = {}
    total = 0

    with open(CATEGORY_STATS_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        # Skip blank lines, separator lines, and header lines
        if not line or "===" in line or "Category" in line or "Total" in line:
            continue

        # Ensure line contains colon separator
        if ": " in line:
            category, count = line.split(": ")
            weights[category] = int(count)
            total += int(count)

    # Normalize weights
    for category in weights:
        weights[category] = weights[category] / total

    return weights

def load_model_results(model, dir_name):
    """Load model evaluation results."""
    file_path = f"./eval/{model}/{dir_name}/combined_output.jsonl"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON from: {file_path}")
        return None

def calculate_metrics(results_data, weights):
    """Calculate weighted average and simple average."""
    if not results_data:
        return None

    weighted_metrics = {
        "weighted": defaultdict(float),
        "average": defaultdict(float)
    }

    category_count = 0

    for category, metrics in results_data.items():
        if category not in weights:
            continue

        category_count += 1
        weight = weights.get(category, 0)

        # Process simple metrics
        for metric in METRICS:
            if metric in metrics:
                weighted_metrics["weighted"][metric] += metrics[metric] * weight
                weighted_metrics["average"][metric] += metrics[metric] / len(results_data)

        # Process metrics under gpt_score
        if "gpt_score" in metrics:
            for gpt_metric in GPT_METRICS:
                if gpt_metric in metrics["gpt_score"]:
                    metric_key = f"gpt_{gpt_metric}"
                    weighted_metrics["weighted"][metric_key] += metrics["gpt_score"][gpt_metric] * weight
                    weighted_metrics["average"][metric_key] += metrics["gpt_score"][gpt_metric] / len(results_data)

    return dict(weighted_metrics)

def main():
    # Load category weights
    category_weights = load_category_weights()
    print(f"Loaded category weights for {len(category_weights)} categories")

    # Result storage
    results = {}

    # Iterate over all models and directories
    for model in MODELS:
        results[model] = {}

        for dir_name in DIRS:
            # Load model results
            model_results = load_model_results(model, dir_name)

            if model_results:
                # Compute metrics
                metrics = calculate_metrics(model_results, category_weights)
                if metrics:
                    results[model][dir_name] = metrics
                    print(f"Processed {model}/{dir_name}")
                else:
                    print(f"No valid metrics found for {model}/{dir_name}")
            else:
                print(f"No valid results found for {model}/{dir_name}")

    # Save results
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
