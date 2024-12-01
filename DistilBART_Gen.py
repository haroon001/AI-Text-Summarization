import torch
from transformers import (
	AutoTokenizer,
	AutoModel,
	AutoModelForSeq2SeqLM,
	AutoModelForCausalLM,
	DataCollatorForSeq2Seq,
	TrainingArguments,
	Trainer,
	Seq2SeqTrainer,
	Seq2SeqTrainingArguments,
	set_seed,
)
from datasets import load_dataset, Dataset, DatasetDict
import evaluate, os
import numpy as np
import pandas as pd


# generate summaries
def generate_summaries(model, tokenizer, test_dataset, num_samples=5):
	model.eval()
	
	selected_indices = range(10)
	
	generation_results = []
	
	with torch.no_grad():
		for idx in selected_indices:
			sample = test_dataset[idx]
			
			input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
			original_text = dataset['test'][idx]['text']
			original_summary = dataset['test'][idx]['summary']
			
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
				'original_summary': original_summary,
				'model_generated_summary': generated_summary
			})
			
	return generation_results

def preprocess_function(examples):
	inputs = ["summarize: " + doc for doc in examples["text"]]
	targets = examples["summary"]
	
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


"""
Load model and tokenizer.
"""
# set seed before initializing model
set_seed(777)
output_path = "./checkpoints"
model_id = os.path.join(output_path, "checkpoint-25070")

# load model
model = AutoModelForSeq2SeqLM.from_pretrained(
	model_id,
	trust_remote_code=True,
	early_stopping=False,
	cache_dir="/scratch0/zamojci1/"
).bfloat16()
model.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

def clean_dataset(dataset):
	df = pd.DataFrame(dataset)
	print(len(df))
	df = df.dropna()
	print(len(df))
	return Dataset.from_pandas(df)

# load dataset
dataset = load_dataset("gursi26/wikihow-cleaned")["train"]
dataset = clean_dataset(dataset)

# split dataset
a = dataset.train_test_split(test_size=0.25)
b = a['test'].train_test_split(test_size=0.5)
dataset = DatasetDict({
	'train': a['train'],
	'test': b['test'],
	'valid': b['train']
})
print(dataset)


# apply tokenization
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# run
generated_samples = generate_summaries(model, tokenizer, tokenized_datasets['test'])

for sample in generated_samples:
	print("\n--- Sample ---")
	print("Original Headline:", sample['original_summary'])
	print("\nOriginal Text (first 500 chars):", sample['original_text'][:500])
	print("\nGenerated Summary:", sample['model_generated_summary'])
	print("-" * 50)