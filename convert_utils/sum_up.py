import json
from collections import Counter

# File paths
input_file = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/sorted_combined_output.jsonl"
output_file = "/home/xiuying.chen/qian_jiang/AudioJailbreak/convert/category_statistics.txt"

# Count entries per category
def count_categories(file_path):
    categories = []

    # Read JSONL file, extract category field
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'category' in data:
                    categories.append(data['category'])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:50]}...")
                continue

    # Use Counter to count entries per category
    category_counts = Counter(categories)
    return category_counts

# Write results to text file
def write_statistics(counts, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Category statistics:\n")
        f.write("=" * 30 + "\n")

        # Sort by count in descending order
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Write each category and its count
        for category, count in sorted_counts:
            f.write(f"{category}: {count}\n")

        # Write total
        f.write("=" * 30 + "\n")
        f.write(f"Total: {sum(counts.values())}\n")

def main():
    print("Starting category count...")
    category_counts = count_categories(input_file)

    print("Writing statistics to file...")
    write_statistics(category_counts, output_file)

    print(f"Statistics complete, results saved to {output_file}")

if __name__ == "__main__":
    main()
