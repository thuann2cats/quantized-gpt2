import re
import string
from collections import Counter
from typing import List, Dict, Tuple
import torch


def normalize_answer(s: str) -> str:
    """
    Normalize answer string for fair comparison.
    
    Steps:
    1. Lowercase
    2. Remove articles (a, an, the)
    3. Remove punctuation
    4. Remove extra whitespace
    
    Args:
        s: Answer string
    
    Returns:
        Normalized string
    
    Example:
        >>> normalize_answer("The Quick Brown Fox!")
        'quick brown fox'
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute Exact Match score.
    
    Binary metric: 1.0 if normalized strings match exactly, else 0.0
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.
    
    F1 = 2 × (precision × recall) / (precision + recall)
    
    Where:
    - precision = |pred ∩ truth| / |pred|
    - recall = |pred ∩ truth| / |truth|
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
    
    Returns:
        F1 score between 0.0 and 1.0
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    # Handle empty predictions
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(len(prediction_tokens) == len(ground_truth_tokens))
    
    # Count token overlap using Counter (handles duplicates)
    prediction_counter = Counter(prediction_tokens)
    ground_truth_counter = Counter(ground_truth_tokens)
    
    # Intersection: min count for each token
    overlap = prediction_counter & ground_truth_counter
    num_same = sum(overlap.values())
    
    if num_same == 0:
        return 0.0
    
    # Precision and recall
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    
    # F1 score
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def extract_answer_from_logits(
    start_logits: torch.Tensor,  # Shape: [batch_size, seq_len]
    end_logits: torch.Tensor,    # Shape: [batch_size, seq_len]
    input_ids: torch.Tensor,     # Shape: [batch_size, seq_len]
    tokenizer,
    max_answer_length: int = 30
) -> List[str]:
    """
    Extract answer spans from start/end logits.
    
    For each example in batch:
    1. Find best span [i, j] that maximizes start_logits[i] + end_logits[j]
       subject to: i <= j and (j - i + 1) <= max_answer_length
    2. Extract tokens: input_ids[i:j+1]
    3. Decode to text
    
    Args:
        start_logits: Model predictions for start positions [batch_size, seq_len]
        end_logits: Model predictions for end positions [batch_size, seq_len]
        input_ids: Token IDs [batch_size, seq_len]
        tokenizer: Tokenizer to decode token IDs
        max_answer_length: Maximum allowed answer length (in tokens)
    
    Returns:
        List of answer strings (one per example in batch)
    """
    batch_size, seq_len = start_logits.shape
    answers = []
    
    # Move to CPU for processing
    start_logits = start_logits.detach().cpu()
    end_logits = end_logits.detach().cpu()
    input_ids = input_ids.detach().cpu()
    
    for batch_idx in range(batch_size):
        # Get logits for this example
        start_log = start_logits[batch_idx]
        end_log = end_logits[batch_idx]
        ids = input_ids[batch_idx]
        
        # Find best span
        best_score = float('-inf')
        best_start = 0
        best_end = 0
        
        for i in range(seq_len):
            for j in range(i, min(i + max_answer_length, seq_len)):
                score = start_log[i].item() + end_log[j].item()
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = j
        
        # Extract answer tokens
        answer_ids = ids[best_start:best_end + 1]
        
        # Decode to text
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        
        # Handle empty answers
        if not answer_text.strip():
            answer_text = ""
        
        answers.append(answer_text)
    
    return answers


def compute_squad_metrics(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    tokenizer
) -> Dict[str, float]:
    """
    Compute SQuAD metrics for a batch.
    
    Args:
        start_logits: Model predictions for start positions [batch_size, seq_len]
        end_logits: Model predictions for end positions [batch_size, seq_len]
        batch: Dict containing:
            - 'input_ids': Token IDs [batch_size, seq_len]
            - 'answers': List of ground truth answer strings
        tokenizer: Tokenizer to decode token IDs
    
    Returns:
        Dict with average metrics across batch:
            {'em': float, 'f1': float}

    """
    # Extract predictions
    predictions = extract_answer_from_logits(
        start_logits,
        end_logits,
        batch['input_ids'],
        tokenizer
    )
    
    # Get ground truth answers
    ground_truths = batch['answers']
    
    # Compute metrics for each example
    em_scores = []
    f1_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        # Handle multiple ground truth answers (SQuAD v2 feature)
        if isinstance(gt, list):
            # Take max score across all ground truths
            em = max(compute_exact_match(pred, g) for g in gt)
            f1 = max(compute_f1_score(pred, g) for g in gt)
        else:
            em = compute_exact_match(pred, gt)
            f1 = compute_f1_score(pred, gt)
        
        em_scores.append(em)
        f1_scores.append(f1)
    
    # Average across batch
    return {
        'em': sum(em_scores) / len(em_scores) if em_scores else 0.0,
        'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    }