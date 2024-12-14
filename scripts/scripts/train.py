import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

def train_model(model_name, train_path, val_path, output_dir, num_labels=10, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    
    train_dataset = load_dataset('json', data_files=train_path)['train']
    val_dataset = load_dataset('json', data_files=val_path)['train']
    
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)
    
    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--train_path", required=True, help="Path to training data")
    parser.add_argument("--val_path", required=True, help="Path to validation data")
    parser.add_argument("--output_dir", required=True, help="Path to save the model")
    args = parser.parse_args()
    train_model(args.model_name, args.train_path, args.val_path, args.output_dir)
