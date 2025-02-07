# General Description of the Repository

This repository contains scripts designed to process and analyze a dataset, **amadeus-dataset-v6.json**, created from the Amadeus API documentation. The dataset focuses on travel-related APIs (such as flight booking and recommendations) and is used for evaluating syntactic and semantic similarity in paraphrased sentences. The repository includes various scripts for processing the data, detecting outliers in paraphrase generation, and evaluating similarity scores using different approaches.

## Key Scripts:
- **`process_amadeus.py`**: Processes the **amadeus-dataset-v6.json** file to compute syntactic similarity scores between each utterance and its generated paraphrases, helping to identify the best prompts for diverse syntax generation.
- **`bert_score_test.py`**: Computes semantic similarity between sentences using the **BERTScore** library, which measures the alignment between candidate and reference sentences on a scale from 0 to 1.
- **`outlier_detection.py`**: Implements paraphrase outlier detection, using the method from the paper [Outlier Detection for Improved Data Quality and Diversity in Dialog Systems](https://aclanthology.org/N19-1051.pdf) to identify errors and unique paraphrases.
- **`print_amadeus.py`**: A utility script for extracting and displaying specific fields from the **amadeus-dataset-v6.json** file, such as prompts and generated utterances.
- **`render_processed_data.py`**: Renders and prettifies the processed JSON data from **process_amadeus.py**, extracting key values like **new syntax templates** and **mean edit distance** for each API endpoint.

The repository provides the tools to analyze the quality and diversity of paraphrases generated for various travel-related APIs and helps in refining the process for improving paraphrase generation models.
