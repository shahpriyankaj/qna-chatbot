from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from datasets import Dataset
import nltk
import sqlite3
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from evaluate import load
# Load the ROUGE metric
import evaluate
rouge = evaluate.load('rouge')

def evaluate_bleu(predictions, references):
    if len(references) != len(predictions):
        raise ValueError("The number of reference texts must match the number of predicted texts.")
    
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
    #weights = (0.25, 0.25, 0, 0)  # Weights for uni-gram, bi-gram, tri-gram, and 4-gram
    #bleu = nltk.translate.bleu_score.sentence_bleu(ref, pred, weights=weights)
    # Assuming predictions and references are lists of text
    #bleu_scores = [nltk.translate.bleu_score.sentence_bleu([ref.split()], pred.split()) for pred, ref in zip(predictions, references)]
    #bleu_scores = [nltk.translate.bleu_score.corpus_bleu([ref.split()], pred.split()) for pred, ref in zip(predictions, references)]
    #avg_bleu = sum(bleu_scores) / len(bleu_scores)
    # Initialize an empty list to store BLEU scores
    bleu_scores = []

    # Loop through each pair of predicted and reference sentences
    for pred, ref in zip(predictions, references):
        #ref_tokenized = [ref.split()]  # Tokenize reference and nest in a list
        #pred_tokenized = pred.split()  # Tokenize prediction
        #print(ref_tokenized)
        #print(pred_tokenized)
        sacrebleu = evaluate.load("sacrebleu")
        score = sacrebleu.compute(predictions=predictions, references=references)
        #smoother = SmoothingFunction().method1
        #score = sentence_bleu(ref_tokenized, pred_tokenized, smoothing_function=smoother)  # Compute BLEU score
        #score = corpus_bleu(ref_tokenized, pred_tokenized)
        bleu_scores.append(round(score["score"], 1))  # Store the result

    return bleu_scores


def evaluate_rouge(predictions, references, rouge_types=('rouge1', 'rougeL')):
    results = rouge.compute(predictions=predictions, references=references)
    scores = []

    # Iterate over the results and store the scores for each prediction
    '''
    for i in enumerate(zip(predictions, references)):      
        scores.append(results['rouge1'][i].fmeasure ) # Access per-row F1 score
    '''
    from rouge_score import rouge_scorer
    """
    Calculates ROUGE scores for each record in a list of reference and predicted texts.

    Args:
        reference_texts: A list of reference texts (ground truth).
        predicted_texts: A list of predicted texts.
        rouge_types: A tuple of ROUGE types to calculate (e.g., 'rouge1', 'rougeL').

    Returns:
        A list of dictionaries, where each dictionary contains the ROUGE scores 
        for a single record.
    """
    if len(references) != len(predictions):
        raise ValueError("The number of reference texts must match the number of predicted texts.")

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    all_scores = []

    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        all_scores.append(scores)
    
    return all_scores
    #return results


def evaluate_accuracy(predictions, references):
    """
    Calculate accuracy by comparing predictions with gold-standard answers.
    """
    if len(references) != len(predictions):
        raise ValueError("The number of reference texts must match the number of predicted texts.")
    return accuracy_score(references, predictions)


def evaluate_raga(questions, answers, contexts, ground_truths):
    data = {
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset = dataset, 
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            ],
        )

    raga_scores = result.to_pandas()
    return raga_scores


def measure_response_time(start_time, end_time):
    """
    Measure the response time of the chatbot for a given query.
    """
    return end_time - start_time


def calculate_user_satisfaction_score():
    """
    Measure the user satisfaction score
    """

    # Connect to SQLite database
    conn = sqlite3.connect('qna_interactions.db')
    c = conn.cursor()

    # Fetch all interactions with thumbs ratings
    c.execute("SELECT thumbs FROM interactions")
    rows = c.fetchall()

    # Calculate the satisfaction score
    thumbs_up_count = sum(1 for row in rows if row[0] == 'thumbs_up')
    thumbs_down_count = sum(1 for row in rows if row[0] == 'thumbs_down')
    total_responses = thumbs_up_count + thumbs_down_count

    # Avoid division by zero
    if total_responses > 0:
        satisfaction_score = (thumbs_up_count / total_responses) * 100
    else:
        satisfaction_score = 0
    conn.close()
    return satisfaction_score