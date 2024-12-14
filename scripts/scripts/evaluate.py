from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from sklearn.metrics import classification_report

def evaluate_model(model_name, test_path, num_labels=10, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    test_dataset = load_dataset('json', data_files=test_path)['train']

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)

    test_dataset = test_dataset.map(tokenize, batched=True)

    predictions, true_labels = [], []
    model.eval()

    with torch.no_grad():
        for batch in test_dataset:
            inputs = torch.tensor(batch['input_ids']).unsqueeze(0)
            outputs = model(inputs).logits
            predictions.extend(torch.argmax(outputs, dim=2).tolist())
            true_labels.extend(batch['labels'])

    print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--test_path", required=True, help="Path to test data")
    args = parser.parse_args()
    evaluate_model(args.model_name, args.test_path)
