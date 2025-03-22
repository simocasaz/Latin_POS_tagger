import os
from collections import Counter

# Analyze the POS tags and tokens in the training data
data_dir = "data/train"

# Extract all the tokens and labels from the training data
files = files = [f for f in os.listdir(data_dir) if f.endswith(".conllu")]

labels = []
tokens = []

for file in files:
    file_path = os.path.join(data_dir, file)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):  # Skip headers and empty lines
                continue
            columns = line.strip().split("\t")
            label = columns[3]
            token_id = columns[0]
            token = columns[1]
            if "-" not in token_id:  # Skip multi-word tokens
                labels.append(label)
                tokens.append(token)

# Extract all the tokens and labels from the validation data
data_dir = "data/validation"

files = files = [f for f in os.listdir(data_dir) if f.endswith(".conllu")]
for file in files:
    file_path = os.path.join(data_dir, file)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):  # Skip headers and empty lines
                continue
            columns = line.strip().split("\t")
            label = columns[3]
            token_id = columns[0]
            token = columns[1]
            if "-" not in token_id:  # Skip multi-word tokens
                labels.append(label)
                tokens.append(token)

# Check for tokens that appears only once
print(len(tokens))
token_counts = Counter(tokens)
single_tokens = [token for token, count in token_counts.items() if count == 1]
print(len(single_tokens))

x_tokens = []
for i in range(len(tokens)):
    token = tokens[i]
    label = labels[i]
    if label == "NUM":
        x_tokens.append((token, label))

print((x_tokens))


label_counts = Counter(labels)
print(label_counts)

import matplotlib.pyplot as plt
from collections import Counter

# Extract labels and frequencies
labels, frequencies = zip(*sorted(label_counts.items(), key=lambda x: x[1], reverse=True))

# Plot
plt.figure(figsize=(12, 8))
plt.bar(labels, frequencies, color='royalblue')

# Labels and title
plt.xlabel("POS Labels")
plt.ylabel("Frequencies")
plt.title("Distribution of POS Labels in the Dataset")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()

# Test data analysis

# Extracting the label from the test files for the confusion matrix

def extract_labels(file_path):
    """
    Extracts the gold labels from a CoNLL-U file.

    Args:
    - gold_file_path (str): Path to the gold file.

    Returns:
    - gold_labels (List[List[str]]): List of sentences, each containing the gold POS tags for each word.
    """
    gold_labels = []
    with open(file_path, 'r', encoding='utf-8') as infile:
        current_sentence = []
        for line in infile:
            if line.startswith("#"):
                continue
            elif line.strip() == "":
                gold_labels.append(current_sentence)
                current_sentence = []
            else:
                parts = line.strip().split("\t")
                if "-" in parts[0]:
                    continue
                current_sentence.append(parts[3])
    return gold_labels

# Extract the gold labels from the gold files
total_gold_labels = []

for file in os.listdir("gold_data"):
    gold_labels = extract_labels(f"gold_data/{file}")
    print(f"Gold labels for {file}: {gold_labels[:2]}")
    total_gold_labels.extend(gold_labels)

# Extract the labels from the test predictions
total_predicted_labels = []

for file in os.listdir("output"):
    if file.endswith(".conllu"):
        test_labels = extract_labels(f"output/{file}")
        print(f"Test labels for {file}: {test_labels[:2]}")
        total_predicted_labels.extend(test_labels)

# Check the length of the gold and test labels
print(len(gold_labels), len(test_labels))

# Create a confusion matrix
from sklearn.metrics import confusion_matrix

# Flatten the gold and test labels
gold_labels_flat = [label for sentence in total_gold_labels for label in sentence]
test_labels_flat = [label for sentence in total_predicted_labels for label in sentence]

# Create the confusion matrix
conf_matrix = confusion_matrix(gold_labels_flat, test_labels_flat)

# Display the confusion matrix
print(conf_matrix)

# Plot the confusion matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Get the unique labels
labels = sorted(set(gold_labels_flat))

# Create a heatmap for the errors in the confusion matrix
# Create a mask to hide the diagonal (correct predictions)
mask = np.eye(len(labels), dtype=bool)  # True for correct predictions

plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Reds", 
            mask=mask, linewidths=0.5, linecolor='gray', cbar=True)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Highlighting Errors")
plt.show()



