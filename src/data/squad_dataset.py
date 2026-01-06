"""
SQuAD v2 dataset loading and preprocessing for question answering.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from transformers import GPT2TokenizerFast
from typing import Dict, List, Tuple, Optional, Any
import random


def load_squad_dataset(dataset_name = 'squad_v2', cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Load SQuAD v2 dataset from HuggingFace.
    
    Args:
        cache_dir: Directory to cache downloaded dataset (uses HF_HOME from .env if None)
    
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return dataset


def preprocess_squad_for_qa(
    examples: Dict[str, List],
    tokenizer: GPT2TokenizerFast,
    max_length: int = 384,
    doc_stride: int = 128
) -> Dict[str, List]:
    """
    Preprocess SQuAD examples for question answering.
    
    1. GPT-2 doesn't have a pad token (use eos_token)
    2. Answer spans must be mapped to token positions
    3. Unanswerable questions (SQuAD v2) → start=0, end=0
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: GPT-2 tokenizer
        max_length: Maximum sequence length
        doc_stride: Stride for sliding window (if context too long)
    
    Returns:
        Dict with input_ids, attention_mask, start_positions, end_positions
    """
    questions_unstripped = examples['question']
    contexts = examples['context']
    answers = examples['answers']
    questions = [q.strip() for q in questions_unstripped]
    
    # Tokenize questions and contexts together
    tokenized = tokenizer(
        questions,
        contexts,
        truncation='only_second',  # Only truncate context, not question
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
        return_tensors=None  # Return lists, not tensors (for batching)
    )
    
    # Map answer character spans to token positions
    offset_mapping = tokenized.pop('offset_mapping')
    sample_mapping = tokenized.pop('overflow_to_sample_mapping')
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        # Get the sample this feature comes from
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]
        
        # Get sequence IDs to identify which tokens are question (0) vs context (1)
        input_ids = tokenized['input_ids'][i]
        sequence_ids = tokenized.sequence_ids(i)
        
        # Find start and end of context in the token sequence
        context_start = None
        context_end = None
        for idx, seq_id in enumerate(sequence_ids):
            if seq_id == 1:  # Context tokens have sequence_id = 1
                if context_start is None:
                    context_start = idx
                context_end = idx
        
        # If no answer (unanswerable question), set positions to 0
        if len(answer['answer_start']) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        # Get answer start character position (relative to context)
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        
        # Find token positions that contain the answer
        # ONLY search within context tokens (between context_start and context_end)
        token_start = None
        token_end = None
        
        for token_idx in range(context_start, context_end + 1):
            offset_start, offset_end = offsets[token_idx]
            
            # Skip special tokens (offset = (0, 0))
            if offset_start == 0 and offset_end == 0:
                continue
            
            # Check if this token contains the answer start
            if token_start is None and offset_start <= start_char < offset_end:
                token_start = token_idx
            
            # Check if this token contains the answer end
            if offset_start < end_char <= offset_end:
                token_end = token_idx
                break
        
        # If answer not found in this context window, set to 0 (treat as unanswerable)
        # This happens when the answer is in a truncated portion
        if token_start is None or token_end is None:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(token_start)
            end_positions.append(token_end)
    
    tokenized['start_positions'] = start_positions
    tokenized['end_positions'] = end_positions

    # Map answers to overflowed examples using sample_mapping
    answers_for_overflow = []
    for i in range(len(tokenized['input_ids'])):
        sample_idx = sample_mapping[i]
        answers_for_overflow.append(answers[sample_idx]['text'])

    tokenized['answers'] = answers_for_overflow

    tokenized['start_positions'] = start_positions
    tokenized['end_positions'] = end_positions
    
    return tokenized



def setup_gpt2_tokenizer_for_qa(model_name = 'gpt2') -> GPT2TokenizerFast:
    """
    Setup GPT-2 tokenizer for question answering.
    
    GPT-2 doesn't have a pad token by default!
    We use eos_token as pad_token.
    
    Returns:
        Configured GPT2TokenizerFast
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    
    # GPT-2 doesn't have pad token, use eos_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # For QA, pad on the right
    tokenizer.padding_side = 'right'
    
    return tokenizer


def create_train_val_split(
    dataset: DatasetDict,
    val_size: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split train set into finetune and validation sets.
    
    Args:
        dataset: DatasetDict with 'train' split
        val_size: Fraction for validation (default: 0.1 = 10%)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (finetune_dataset, val_dataset)
    """
    train_dataset = dataset['train']
    
    split = train_dataset.train_test_split(test_size=val_size, seed=seed)
    
    return split['train'], split['test']  # 'train' = finetune, 'test' = validation


class SQuADDataset(Dataset):
    """
    PyTorch Dataset wrapper for preprocessed SQuAD data.
    """
    
    def __init__(self, hf_dataset: Dataset, tokenizer: GPT2TokenizerFast, max_length: int = 384):
        """
        Args:
            hf_dataset: HuggingFace Dataset (already split)
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Preprocess all examples
        self.features = self.dataset.map(
            lambda examples: preprocess_squad_for_qa(examples, tokenizer, max_length),
            batched=True,
            # remove_columns=self.dataset.column_names,
            remove_columns=self.dataset.column_names,
            desc="Preprocessing SQuAD"
        )

    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example as tensors.
        
        Returns:
            Dict with 'input_ids', 'attention_mask', 'start_positions', 'end_positions'
        """
        item = self.features[idx]
        
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'start_positions': torch.tensor(item['start_positions'], dtype=torch.long),
            'end_positions': torch.tensor(item['end_positions'], dtype=torch.long),
            'answers': item['answers'],
        }        
    

def collate_squad_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for SQuAD batches.
    - Stacks tensors for input_ids, attention_mask, etc.
    - Collects 'answers' into a list (because list lengths vary per example)
    """
    # Separate answers from tensor fields
    answers = [item.pop('answers') for item in batch]
    
    # Use default_collate for remaining tensor fields
    from torch.utils.data.dataloader import default_collate
    collated_batch = default_collate(batch)
    
    # Add answers back as a simple list
    collated_batch['answers'] = answers
    
    return collated_batch


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create PyTorch DataLoader.
    
    Args:
        dataset: SQuADDataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
    
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_squad_batch  # Custom collate function
    )
    

def prepare_squad_data(
    dataset_name: str = "squad_v2",
    batch_size: int = 32,
    max_length: int = 384,
    val_size: float = 0.1,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, GPT2TokenizerFast]:
    """
    One-stop function to prepare all SQuAD data.
    
    Args:
        batch_size: Batch size
        max_length: Max sequence length
        val_size: Validation split size (0.1 = 10%)
        cache_dir: Cache directory for dataset
    
    Returns:
        Tuple of (train_dataloader, val_dataloader, tokenizer)
    """
    # Load dataset
    print("Loading SQuAD v2 dataset...")
    dataset = load_squad_dataset(dataset_name, cache_dir)
    
    # Setup tokenizer
    print("Setting up tokenizer...")
    tokenizer = setup_gpt2_tokenizer_for_qa()
    
    # Split train into finetune + val
    print(f"Splitting train set ({val_size*100:.0f}% for validation)...")
    finetune_dataset, val_dataset = create_train_val_split(dataset, val_size)
    
    # Create PyTorch datasets
    print("Creating PyTorch datasets...")
    finetune_squad = SQuADDataset(finetune_dataset, tokenizer, max_length)
    val_squad = SQuADDataset(val_dataset, tokenizer, max_length)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(finetune_squad, batch_size, shuffle=True)
    val_loader = create_dataloader(val_squad, batch_size, shuffle=False)
    
    print(f"✅ Data ready! Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, tokenizer