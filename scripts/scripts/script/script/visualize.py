import matplotlib.pyplot as plt
import json

def visualize_performance(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    genres = list(metrics.keys())
    f1_scores = [metrics[genre]['f1'] for genre in genres]

    plt.bar(genres, f1_scores, color='skyblue')
    plt.title('Genre-wise F1-Score')
    plt.xlabel('Genres')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", required=True, help="Path to JSON metrics file")
    args = parser.parse_args()
    visualize_performance(args.metrics_path)
