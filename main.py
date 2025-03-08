from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from sklearn.model_selection import train_test_split
import os, re
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader


## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set_theme('notebook', style='whitegrid')
from tqdm.notebook import tqdm

# Gpu check
print("Gpu check: started")
# Set the seed for reproducibility
torch.manual_seed(42)

print("Using torch", torch.__version__)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

if torch.cuda.is_available():
    x = torch.ones(1, device=device)
    print(x)
    
    # GPU operations have a separate seed we also want to set
    torch.cuda.manual_seed(42)

# Print CUDA availability and version
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

print("Gpu check: completed")

# Path to the dataset directory
DATA_DIR = "data"

allowed_classes = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "VERB", "X"]

# Create the label_to_id dictionary by enumerating the allowed_classes
label_to_id = {label: idx for idx, label in enumerate(allowed_classes)}

class PosDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        """
        Initialize the dataset by loading all the files in the specified split (train/test).
        
        Args:
        - data_dir (str): The directory where the dataset is stored.
        - split (str): 'train' or 'test'.
        """
        self.data_dir = os.path.join(data_dir, split)
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith(".conllu")]
        self.data = []
        self._load_data()

    def _load_data(self):
        """ Load and parse the data from the files. """
        for file in self.files:
            file_path = os.path.join(self.data_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                tokens, labels = [], []
                for line in f:
                    if line.strip() == "" or line.startswith("#"):  # Skip headers and empty lines
                        continue
                    columns = line.strip().split("\t")
                    token = columns[1]  # The word/token itself
                    label = columns[3] # The POS label
                    if label != "_":
                        tokens.append(token)
                        labels.append(label)

                self.data.append((tokens, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the tokens and labels for the given index.
        Args:
        - idx (int): The index of the article.
        Returns:
        - tokens (List[str]): The list of tokens.
        - labels (List[str]): The corresponding NER labels for the tokens.
        """
        tokens, labels = self.data[idx]
        return tokens, labels

def collate_fn(batch, tokenizer, label_to_id, max_length=512, label_pad_token=-100):
    """
    Collate function to process batches with tokens and labels stored in tuples.
    
    Args:
    - batch: List of dicts containing 'tokens' and 'labels', where both are tuples.
    - tokenizer: Pretrained tokenizer (RobertaTokenizer).
    - label_to_id: Dictionary mapping string labels (e.g., 'LOC') to integers.
    - max_length: Maximum sequence length (default 512 for RoBERTa).
    - label_pad_token: Token to pad labels, default is -100.
    
    Returns:
    - input_ids: Tensor of tokenized input sequences.
    - labels: Tensor of padded labels.
    - attention_mask: Tensor mask to ignore padding tokens.
    """
    
    # Extract tokens and labels from tuples
    tokens = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Tokenize the tokens with truncation and padding
    tokenized_inputs = tokenizer(tokens, 
                                 is_split_into_words=True, 
                                 padding=True, 
                                 truncation=True, 
                                 max_length=max_length,  # Ensure max length, also drops what is longer!
                                 return_tensors="pt",
                                 return_offsets_mapping=True)
    
    # Create aligned labels based on the word_ids.
    all_labels = []
    # We loop through each example in the batch.
    for i, label_seq in enumerate(labels):
        # Get mapping from tokens to original word indices.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special token (like CLS, SEP, or padding)
                label_ids.append(label_pad_token)
            elif word_idx != previous_word_idx:
                # First token of a given word: assign its label.
                label_ids.append(label_to_id.get(label_seq[word_idx], label_pad_token))
            else:
                # Subsequent subword token: assign pad token so it's ignored in loss.
                label_ids.append(label_pad_token)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    
    # Remove offset mapping if not needed.
    tokenized_inputs.pop("offset_mapping", None)
    
    # Convert label lists into a tensor.
    padded_labels = torch.tensor(all_labels, dtype=torch.long)
    
    return tokenized_inputs['input_ids'], padded_labels, tokenized_inputs['attention_mask']

# Load the dataset
train_dataset = PosDataset(DATA_DIR, split="train")

