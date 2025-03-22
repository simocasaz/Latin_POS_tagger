# Latin_POS_tagger
## Introduction
This project focuses on developing a Part-of-Speech (POS) tagging system for Ancient Latin. The system is inspired by the [approach](https://aclanthology.org/2022.lt4hala-1.31.pdf) used by the winning team in the EvaLatin 2022 challenge and aims to utilize the XML-RoBERTa base model for high-quality POS tagging in Latin texts.

## Getting Started
The repository includes two versions of the main script. The first is a Jupyter notebook adapted for use with Google Colab. To run this version, ensure that you mount your Google Drive and update the path variables to reflect your personalized directory structure before executing the script. The second version is a Python script that reads data from local files. In this case, make sure to fill in the path variables with your local file paths before running the script.

Additionally, there are several other scripts included in the repository:

- `data-analysis.py` is used to analyze the data prior to training and to investigate the model's errors.
- `conll18_ud_eval_EvaLatin_2022.py` contains the original scoring script and is located in the `scorer` directory.
- `score.py` imports the scorer to evaluate the test files.

The data is organized as it was for the training process, and the `output` directory contains the results of the final tests.

## For More Information
For further details on the project, please refer to the [project report](Casazza_ML_report.pdf). Additional useful resources are listed below:

- [Link to the EvaLatin Dataset](https://github.com/CIRCSE/LT4HALA/tree/master/2022/data_and_doc)
- [Link to the Paper on the Evaluation Campaign and Results](https://aclanthology.org/2022.lt4hala-1.29.pdf)