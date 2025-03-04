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
from torch.optim import AdamW


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

# Load the dataset
train_dataset = PosDataset(DATA_DIR, split="train")

print(train_dataset[0])