import os
from collections import Counter

data_dir = "data/train"

files = files = [f for f in os.listdir(data_dir) if f.endswith(".conllu")]

labels = []

for file in files:
    file_path = os.path.join(data_dir, file)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):  # Skip headers and empty lines
                continue
            columns = line.strip().split("\t")
            label = columns[3]
            if label != "_":
                labels.append(label)

label_counts = Counter(labels)
print(label_counts)