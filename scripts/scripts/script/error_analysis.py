import json
from collections import Counter

def analyze_errors(predictions_path, labels_path):
    with open(predictions_path, 'r') as pred_file, open(labels_path, 'r') as label_file:
        predictions = json.load(pred_file)
        labels = json.load(label_file)

    errors = [pred for pred, label in zip(predictions, labels) if pred != label]
    error_counts = Counter(errors)
    print("Error Breakdown:", error_counts)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True, help="Path to predictions JSON file")
    parser.add_argument("--labels_path", required=True, help="Path to ground truth labels JSON file")
    args = parser.parse_args()
    analyze_errors(args.predictions_path, args.labels_path)
