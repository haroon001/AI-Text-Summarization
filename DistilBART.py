import torch
import logging
import transformers
import sys
import os
import nltk
import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
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
from transformers.trainer_utils import get_last_checkpoint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge


nltk.download('punkt')
nltk.download('punkt_tab')
meteor = evaluate.load("meteor")

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def calculate_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores for a given reference and hypothesis.

    :param reference: The reference (ground truth) summary
    :param hypothesis: The generated summary to evaluate
    :return: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return {
        'rouge-1': scores[0]['rouge-1']['f'],
        'rouge-2': scores[0]['rouge-2']['f'],
        'rouge-l': scores[0]['rouge-l']['f']
    }

def calculate_meteor(decoded_labels, decoded_preds):
    """
    Calculate the METEOR score for a given predictions and labels.
    
    :param decoded_preds (list): Decoded predictions of model. 
    :param decoded_labels (list): Decoded or original labels.
    :return (float): The METEOR score
    """
    try:
        meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
        
    except Exception as e:
        print(f"Warning: Error computing METEOR score: {str(e)}")
        meteor_result = 0.0

    return meteor_result

def calculate_bleu(reference, hypothesis):
    """
    Calculate the BLEU score for a given reference and hypothesis.

    :param reference (str): The reference (ground truth) summary
    :param hypothesis (str): The generated summary to evaluate
    :return (float): The BLEU score
    """

    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    sf = SmoothingFunction()
    return sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=sf.method1)

def evaluate_summary(reference, hypothesis):
	"""
	Evaluate a generated summary using both ROUGE and BLEU metrics.

	:param reference (str): The reference (ground truth) summary
	:param hypothesis (str): The generated summary to evaluate
	:return (float): A dictionary containing ROUGE and BLEU scores
	"""
	rouge_scores = calculate_rouge(reference, hypothesis)
	# bleu_score = calculate_bleu(reference, hypothesis)
	meteor_scores = calculate_meteor(reference, hypothesis)
	return {
		'rouge': rouge_scores,
		'meteor': meteor_scores,
		# 'bleu': bleu_score
	}

"""
Load model and tokenizer.
"""
# set seed before initializing model
set_seed(777)

output_path = "./checkpoints"

# model_id = "sshleifer/distilbart-xsum-12-3"
model_id = os.path.join(output_path, "checkpoint-20056")

# load model
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    early_stopping=False,
	cache_dir="/scratch0/zamojci1/"
).bfloat16()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '<pad>'})


"""
Load & clean WikiHow dataset.

https://huggingface.co/datasets/gursi26/wikihow-cleaned
https://github.com/mahnazkoupaee/WikiHow-Dataset
"""
def clean_dataset(dataset):
    df = pd.DataFrame(dataset)
    print(len(df))
    df = df.dropna()
    # df = df.iloc[:100]
    print(len(df))
    return Dataset.from_pandas(df)

def download_dataset():
    # data_path = "./wikihow-cleaned.csv"
    # df = pd.read_csv(data_path)
    dataset = load_dataset("gursi26/wikihow-cleaned")["train"]
    # dataset = Dataset.from_pandas(df)
    return dataset

# load dataset
dataset = download_dataset()
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


# In[7]:


"""
Data tokenization & collation.
"""
# data collator
label_pad_token_id = tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8,
)

max_input_length = 256
max_target_length = 128

if model_id in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples['text']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # setup the tokenizer for targets
    labels = tokenizer(text_target=examples['summary'], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# apply tokenization
tokenized_datasets = dataset.map(preprocess_function, batched=True)


# In[8]:


"""
Evaluation metrics.
"""
def compute_metrics(eval_preds):
	# prepare prediction data
	labels, preds = eval_preds.label_ids, eval_preds.predictions
	labels[labels == -100] = tokenizer.pad_token_id

	# decode
	preds_decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
	labels_decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)

	# calculate rouge scores
	# rouge_scores = calculate_rouge(labels_decoded, preds_decoded)
	scores = evaluate_summary(labels_decoded, preds_decoded)

	# return metrics
	return scores


# In[9]:


"""
Create Trainer.
"""
batch_size = 32
training_args = Seq2SeqTrainingArguments(
    learning_rate=0.005,
    weight_decay=0.01,
    log_level='info',
    output_dir=output_path,
    save_strategy='epoch',
    # save_steps=500,
    save_total_limit=2,
    use_cpu=False,
    eval_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    # fp16=True,
    bf16=True,
)

trainer = Seq2SeqTrainer(
    model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)


# In[ ]:


"""
Train model.
"""
os.environ["WANDB_DISABLED"] = "true"
training_results = trainer.train()
print(training_results)

# In[ ]:


"""
Evaluate model.
"""
testing_results = trainer.evaluate(tokenized_datasets['test'])
print(testing_results)

# In[ ]:


"""
Save trained model.
"""
trainer.save_model(output_path)


"""
Manually generate summaries.
"""
texts = [
	"Caged animals such as hamsters, guinea pigs, rodent, reptiles, and amphibians can be taken to a friend's or sitter's home. Create a document that outlines the feeding and water needs, cleaning schedule, and temperature control. Pack all of the things that mimic your pet's environment at your home such as bedding, heated surfaces, and decorations.If the cage is not mobile, someone will need to come check on your pet daily.A rabbit, ferret, or guinea pig is live bait in the wild. Relocating your",
	"Every boarding barn has people who go on vacation, or people who'd love to take a break from mucking stalls every day. Or even just riding all the time. Set up a rate schedule, depending on how much care you are willing to do. Not only will you get a little bit of income, but you'll learn a lot about horse-ownership along the way.;, Offer to pull manes, braid, polish hooves, etc. If you're boarding your horse in a show barn, you can probably find customers fairly easily. Braiding, especially, c",
	"Let him know that you like him by making an effort to look nice whenever you're around him. You should still be yourself, but take extra care with your hair and makeup and outfits, so he can start to notice you. You don't have to wear a tight dress and high heels if you're at a baseball game with him, but let him know that you care about your looks when you're around him.. Don't be afraid to be a little sexy. If you're comfortable with your body, show it off. If you're not comfortable with a l.",
	"It can be hard to tell if someone is interested, but you should be able to tell if he makes an effort to spend time with you. If he initiates conversation and smiles a lot, then there's a good chance that he is interested. That said: many guys get shy when they like someone, and you shouldn't write someone off just because he hasn't made a move.This can be anything from a trip to the beach to a house party. Some guys like to make the first move, but most will respect a girl who goes for wha.",
]

# model.to('cuda')
model.eval()

selected_indices = range(10)

with torch.no_grad():
	for idx in selected_indices:
		sample = test_dataset[idx]
		
		inputs = tokenizer(
			f"summarize: {text}",
			max_length=512,
			truncation=True,
			return_tensors="pt",
		).to('cuda')
		
		outputs = model.generate(
			inputs.input_ids,
			max_length=128,
			# max_new_tokens=128,
			# do_sample=False,
			# no_repeat_ngram_size=2,
			num_beams=4,
			early_stopping=True,
		)
		decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

		print(text)
		print(decoded)
		print()
