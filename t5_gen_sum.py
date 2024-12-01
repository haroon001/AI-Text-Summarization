import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, DatasetDict
import evaluate, os
from huggingface_hub import login
import numpy as np


login('')

def generate_summaries(model, tokenizer, test_dataset, num_samples=5):
    model.eval()
    
    selected_indices = range(10)
    
    generation_results = []
    
    with torch.no_grad():
        for idx in selected_indices:
            sample = test_dataset[idx]
            
            input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            original_text = dataset_dict['test'][idx]['text']
            original_headline = dataset_dict['test'][idx]['headline']
            
            inputs = tokenizer(
                f"summarize: {input_text}", 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to('cuda')
            
            outputs = model.generate(
                inputs.input_ids, 
                max_length=128, 
                num_beams=4, 
                early_stopping=True
            )
            
            
            generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            
            generation_results.append({
                'original_text': original_text,
                'original_headline': original_headline,
                'model_generated_summary': generated_summary
            })
            
    return generation_results

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["text"]]
    targets = examples["headline"]
    
    
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding="max_length"
        )["input_ids"]
    
    
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]
    
    model_inputs["labels"] = labels
    return model_inputs


model_dir = "./results/checkpoint-11500"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)
model.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

dataset_dict = load_dataset("wikihow", "all", data_dir='.')


tokenized_datasets = DatasetDict({
    split: dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["title", "text", "headline"],
        num_proc=8
    )
    for split, dataset in dataset_dict.items() if split == 'test'
})


generated_samples = generate_summaries(model, tokenizer, tokenized_datasets['test'])

for sample in generated_samples:
    print("\n--- Sample ---")
    print("Original Headline:", sample['original_headline'])
    print("\nOriginal Text (first 500 chars):", sample['original_text'][:500])
    print("\nGenerated Summary:", sample['model_generated_summary'])
    print("-" * 50)