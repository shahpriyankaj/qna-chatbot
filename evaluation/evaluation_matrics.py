''' Script to calculate different kind of evaluation metrics'''
from sklearn.metrics import  accuracy_score
from rouge_score import rouge_scorer
from datasets import Dataset
import sqlite3
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import evaluate

def evaluate_bleu(predictions, references):
    ''' Evaluate bleu score  and returns the row-wise score'''
    if len(references) != len(predictions):
        raise ValueError("The number of reference texts must match the number of predicted texts.")
    
    bleu_scores = []
    # Loop through each pair of predicted and reference sentences
    for pred, ref in zip(predictions, references):
        sacrebleu = evaluate.load("sacrebleu")
        score = sacrebleu.compute(predictions=predictions, references=references)
        bleu_scores.append(round(score["score"], 1)) 

    return bleu_scores


def evaluate_rouge(predictions, references, rouge_types=('rouge1', 'rougeL')):
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
    scores = []
    if len(references) != len(predictions):
        raise ValueError("The number of reference texts must match the number of predicted texts.")

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    all_scores = []

    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        all_scores.append(scores)
    
    return all_scores


def evaluate_accuracy(predictions, references):
    """
    Calculate accuracy by comparing predictions with gold-standard answers.
    """
    if len(references) != len(predictions):
        raise ValueError("The number of reference texts must match the number of predicted texts.")
    return accuracy_score(references, predictions)


def evaluate_raga(questions, answers, contexts, ground_truths):
    '''evaluate raga score and returns the row-wise score'''
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