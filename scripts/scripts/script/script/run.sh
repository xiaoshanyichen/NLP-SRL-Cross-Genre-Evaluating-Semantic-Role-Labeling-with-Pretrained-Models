#!/bin/bash
# End-to-end pipeline script

# Preprocessing
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --tokenizer bert-base-uncased

# Training
python scripts/train.py --model_name bert-base-uncased --train_path data/processed/train.json --val_path data/processed/val.json --output_dir outputs/bert

# Evaluation
python scripts/evaluate.py --model_name outputs/bert --test_path data/processed/test.json

# Error Analysis
python scripts/error_analysis.py --predictions_path outputs/bert/predictions.json --labels_path data/processed/test_labels.json

# Visualization
python scripts/visualize.py --metrics_path outputs/bert/metrics.json
