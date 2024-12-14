import os
import json
from transformers import AutoTokenizer
import argparse

def preprocess_data(input_dir, output_dir, tokenizer_name, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for genre_file in os.listdir(input_dir):
        genre_path = os.path.join(input_dir, genre_file)
        with open(genre_path, 'r') as f:
            data = json.load(f)
        
        tokenized_data = []
        for item in data:
            text = item["sentence"]
            labels = item["labels"]
            tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
            tokenized["labels"] = labels
            tokenized_data.append(tokenized)
        
        output_path = os.path.join(output_dir, genre_file)
        with open(output_path, 'w') as f:
            json.dump(tokenized_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to raw data directory")
    parser.add_argument("--output_dir", required=True, help="Path to save preprocessed data")
    parser.add_argument("--tokenizer", default="bert-base-uncased", help="Tokenizer name")
    args = parser.parse_args()
    preprocess_data(args.input_dir, args.output_dir, args.tokenizer)
