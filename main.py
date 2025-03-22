from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification
from transformers import get_scheduler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.optim import AdamW
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import os

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
BASE_DIR = "" # Put your path here
PERSONALIZED_PATH = "pos_project/data_splitted" # Put your path here
DATA_DIR = os.path.join(BASE_DIR, PERSONALIZED_PATH)
print(DATA_DIR)

allowed_classes = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "VERB", "X"]

# Create the label_to_id dictionary by enumerating the allowed_classes
label_to_id = {label: idx for idx, label in enumerate(allowed_classes)}

class PosDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        print(f"Initializing {split} dataset...")
        self.split = split
        self.data_dir = os.path.join(data_dir, split)
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith(".conllu")]
        self.data = []
        self._load_data()
        print(f"{split} dataset loaded with {len(self.data)} sentences")

    def _load_data(self):
        """ Load and parse the data from the files. """
        for file in self.files:
            file_path = os.path.join(self.data_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                tokens, labels = [], []
                for line in f:
                    if line.startswith("#"):  # Skip headers
                        continue
                    if line.strip() == "":
                        self.data.append((tokens, labels))
                        tokens, labels = [], []
                        continue
                    columns = line.strip().split("\t")
                    token = columns[1]  # The word/token itself
                    label = columns[3] # The POS label
                    if "-" in columns[0]:  # Skip multi-word tokens
                        continue
                    else:
                        tokens.append(token)
                        labels.append(label)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the tokens and labels for the given index.
        Args:
        - idx (int): The index of the sentence.
        Returns:
        - tokens (List[str]): The list of tokens.
        - labels (List[str]): The corresponding POS tag for the tokens.
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

def save_checkpoint(model, optimizer, epoch, loss, file_path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch} to {file_path}")

def load_checkpoint(model, optimizer, file_path="checkpoint.pth"):
    checkpoint = torch.load(file_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    loss = checkpoint['loss']
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    return model, optimizer, start_epoch, loss

def train(model, optimizer, train_loader, epochs, start_epoch, loss_fn, scaler, scheduler=None, accumulation_steps=2, checkpoint_save_path=None):
    print("Starting training...")
    model = model.train()

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        correct_predictions = 0
        total_tokens = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            tokens, labels, mask = batch

            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                outputs = model(input_ids=tokens, attention_mask=mask)
                loss = loss_fn(outputs.logits.view(-1, outputs.logits.shape[-1]), labels.view(-1))
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)

            # Flatten tensors
            predictions_flat = predictions.view(-1)
            labels_flat = labels.view(-1)
            mask_flat = mask.view(-1) == 1  # Boolean mask for valid tokens

            # Exclude padding (-100 labels)
            valid_labels = labels_flat != -100
            valid_mask = mask_flat & valid_labels  # Only count valid non-padding tokens

            # Compute accuracy
            correct_predictions += (predictions_flat[valid_mask] == labels_flat[valid_mask]).sum().item()
            total_tokens += valid_mask.sum().item()

            print(f"Batch {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_epoch_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
        print(f"Epoch {epoch + 1} completed, Average Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        if epoch % 2 == 0 and checkpoint_save_path:
            save_checkpoint(model, optimizer, epoch, loss_fn, checkpoint_save_path)

    print("Training completed")


def evaluate(model, validation_loader, loss_fn):
    print("Starting evaluation...")
    model = model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_tokens = 0

    with torch.no_grad():
        for tokens, labels, mask in validation_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                outputs = model(input_ids=tokens, attention_mask=mask)
                loss = loss_fn(outputs.logits.view(-1, outputs.logits.shape[-1]), labels.view(-1))
                total_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)

                # Flatten tensors
                predictions_flat = predictions.view(-1)
                labels_flat = labels.view(-1)
                mask_flat = mask.view(-1) == 1  # Boolean mask for valid tokens

                # Exclude padding (-100 labels)
                valid_labels = labels_flat != -100
                valid_mask = mask_flat & valid_labels  # Only count valid non-padding tokens

                # Compute accuracy
                correct_predictions += (predictions_flat[valid_mask] == labels_flat[valid_mask]).sum().item()
                total_tokens += valid_mask.sum().item()

    avg_loss = total_loss / len(validation_loader)
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
    print(f"Evaluation completed - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def test(model, test_loader, test_dataset, path_to_test_file, output_path, id_to_label, max_length):
    print("Starting testing...")
    model = model.eval()

    all_sentence_preds = []

    with torch.no_grad():
        j = 0
        # Iterate over the test_loader batches.
        for batch in test_loader:
            # Unpack batch; note: labels may be dummy values since the test set has no gold labels.
            tokens, _, attention_mask = batch  # our collate_fn returns (input_ids, labels, attention_mask)
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=tokens, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (batch_size, seq_length, num_labels)
            predictions = torch.argmax(logits, dim=-1)  # shape: (batch_size, seq_length)

    # Insert the predicted labels into the test file
    # To align predictions with original words, we need to get the word_ids for each sentence.
            # Since our collate_fn discarded the offset mapping, we can re-tokenize the original tokens.
            # We assume that the test dataset provides the original token lists.
            batch_size = tokens.size(0)
            for i in range(batch_size):
                # Retrieve the original token list for sentence i from the dataset.
                # Here, we assume that test_loader.dataset has an attribute "sentences" which is a list of token lists.
                orig_tokens = test_dataset[i+j][0]  # [0] is the tokens, [1] is the labels
                # Re-tokenize with is_split_into_words=True to get the mapping.
                tokenized = tokenizer(orig_tokens, is_split_into_words=True, truncation=True,
                                        max_length=max_length, return_tensors="pt")
                # Get word_ids for the i-th example.
                # (Since we processed one sentence, we use batch_index 0.)
                word_ids = tokenized.word_ids(batch_index=0)

                # Now align the predictions: assign the predicted label of the first subword of each word.
                sentence_pred_ids = predictions[i].tolist()
                aligned_preds = []
                previous_word_idx = None
                token_idx = 0  # pointer into sentence_pred_ids
                for word_idx in word_ids:
                    if word_idx is None:
                        token_idx += 1
                        continue
                    if word_idx != previous_word_idx:
                        # Use the prediction for this token (first subword) for the word.
                        aligned_preds.append(id_to_label[sentence_pred_ids[token_idx]])
                    previous_word_idx = word_ids[token_idx]
                    token_idx += 1
                all_sentence_preds.append(aligned_preds)
            j += batch_size

    # Now, all_sentence_preds is a list of sentences, each containing predicted POS tags for each word.

    # Open the original test file and write a new file with predictions inserted.
    with open(path_to_test_file, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        sentence_idx = 0  # to index into all_sentence_preds
        token_idx = 0     # token index within the current sentence
        for line in infile:
            if line.startswith("#"):
                # Comment or metadata lines are written as-is.
                outfile.write(line)
            elif line.strip() == "":
                # End of a sentence; write blank line and reset token counter.
                outfile.write("\n")
                sentence_idx += 1
                token_idx = 0
            else:
                # Token lines: columns are separated by tabs.
                parts = line.strip().split("\t")
                # In CoNLL-U, the columns are: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC.
                # We assume the predicted POS tag should replace the current UPOS value (column index 3).
                # Handle multi-word tokens (e.g., "12-13") by leaving them unchanged.
                if "-" in parts[0]:
                    outfile.write(line)
                else:
                    # If we have a prediction for this token, replace the underscore in column 4.
                    if sentence_idx < len(all_sentence_preds) and token_idx < len(all_sentence_preds[sentence_idx]):
                        parts[3] = all_sentence_preds[sentence_idx][token_idx]
                    outfile.write("\t".join(parts) + "\n")
                    token_idx += 1

    print(f"Predicted file saved to {output_path}")

    return all_sentence_preds



print("Loading the model...")
model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_to_id))
print("Model loaded")

# Options
BATCH_SIZE = 16
MAX_LENGTH = 256
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

print("Loading the dataset...")
# Load the dataset
train_dataset = PosDataset(DATA_DIR, split="train")
val_dataset = PosDataset(DATA_DIR, split="validation")
print("Dataset loaded")

# Load the tokenizer
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

print("Creating dataloaders...")
# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_fn(batch, tokenizer, label_to_id, max_length=MAX_LENGTH), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_fn(batch, tokenizer, label_to_id, max_length=MAX_LENGTH), shuffle=False)
print("Dataloaders created")

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Scheduler
num_training_steps = len(train_loader) * NUM_EPOCHS
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


# Loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=-100).to(device)

# Scaler
scaler = torch.amp.GradScaler("cuda")

# Ensure all layers are trainable
# NB this is costly in terms of memory and computation
for param in model.parameters():
    param.requires_grad = True

# Move model to device
model = model.to(device)

# Clear CUDA cache before training
torch.cuda.empty_cache()

print("Starting training process...")

CHECKPOINT_PATH = "" # Put here where you want to save your checkpoints
save_path = os.path.join(BASE_DIR, CHECKPOINT_PATH, "checkpoint.pth") # The save path for your checkpoints

# Train the model
if os.path.exists(save_path):
    model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, save_path)
else:
    start_epoch = 0  # Start from scratch

print(save_path)
print(start_epoch)

train(model, optimizer, train_loader, NUM_EPOCHS, start_epoch, loss_fn, scaler, scheduler, 2, save_path)
print("Training process completed")

print("Starting evaluation process...")

# Evaluate the model
val_loss = evaluate(model, val_loader, loss_fn)

print(f"Validation loss and accuracy: {val_loss}")

print("Starting testing process...")

id_to_label = {v: k for k, v in label_to_id.items()}

# Test the model on classical data (Livius)
test_dataset_classic = PosDataset(DATA_DIR, split="test_classical_subtask")
test_loader_classic = DataLoader(test_dataset_classic, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_fn(batch, tokenizer, label_to_id, max_length=MAX_LENGTH), shuffle=False)
test_classical_predictions = (test(model, test_loader_classic, test_dataset_classic, os.path.join(DATA_DIR, "test_classical_subtask/Livius_AbVrbeCondita.conllu"), os.path.join(BASE_DIR, "pos_project/output/test_predictions_livius.conllu"), id_to_label, max_length=MAX_LENGTH))
print("Testing process completed")
print(test_classical_predictions)

# Test the model on cross-genre data (Ovidius)
test_dataset_crossgenre_one = PosDataset(DATA_DIR, split="test_crossgenre_subtask1")
test_loader_crossgenre_one = DataLoader(test_dataset_crossgenre_one, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_fn(batch, tokenizer, label_to_id), shuffle=False)

test(model, test_loader_crossgenre_one, test_dataset_crossgenre_one, os.path.join(DATA_DIR, "test_crossgenre_subtask1/Ovidius_Metamorphoseon.conllu"), os.path.join(BASE_DIR, "pos_project/output/test_predictions_ovidius.conllu"), id_to_label, max_length=MAX_LENGTH)

# Test the model on cross-genre data (Plinius)
test_dataset_crossgenre_two = PosDataset(DATA_DIR, split="test_crossgenre_subtask2")
test_loader_crossgenre_two = DataLoader(test_dataset_crossgenre_two, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_fn(batch, tokenizer, label_to_id), shuffle=False)

test(model, test_loader_crossgenre_two, test_dataset_crossgenre_two, os.path.join(DATA_DIR, "test_crossgenre_subtask2/Plinius_NaturalisHistoria.conllu"), os.path.join(BASE_DIR, "pos_project/output/test_predictions_plinius.conllu"), id_to_label, max_length=MAX_LENGTH)

# Test the model on cross-time data (Sabellicus)
test_dataset_crosstime = PosDataset(DATA_DIR, split="test_crosstime_subtask")
test_loader_crosstime = DataLoader(test_dataset_crosstime, batch_size=BATCH_SIZE, collate_fn=lambda batch: collate_fn(batch, tokenizer, label_to_id), shuffle=False)

test(model, test_loader_crosstime, test_dataset_crosstime, os.path.join(DATA_DIR, "test_crosstime_subtask/Sabellicus_DeLatinaeLinguaeReparatione.conllu"), os.path.join(BASE_DIR, "pos_project/output/test_predictions_sabellicus.conllu"), id_to_label, max_length=MAX_LENGTH)

path_save_model = "" # The path to save the model

model.save_pretrained(os.path.join(BASE_DIR, path_save_model))
tokenizer.save_pretrained(os.path.join(BASE_DIR, path_save_model))

