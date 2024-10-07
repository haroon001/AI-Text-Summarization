import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

nltk.download('punkt')
nltk.download('punkt_tab')

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
    bleu_score = calculate_bleu(reference, hypothesis)
    
    return {
        'rouge': rouge_scores,
        'bleu': bleu_score
    }

# Example usage
if __name__ == "__main__":
    reference = "The quick brown fox jumps over the lazy dog."
    hypothesis = "A fast fox jumps over a lazy dog."
    
    results = evaluate_summary(reference, hypothesis)
    print("Evaluation Results:")
    print(f"ROUGE-1: {results['rouge']['rouge-1']:.4f}")
    print(f"ROUGE-2: {results['rouge']['rouge-2']:.4f}")
    print(f"ROUGE-L: {results['rouge']['rouge-l']:.4f}")
    print(f"BLEU: {results['bleu']:.4f}")