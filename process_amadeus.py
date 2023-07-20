import argparse
import json
from datetime import datetime
import pickle
from itertools import combinations
from typing import Dict, List, Tuple

import stanza
import evalsyntree as evt

"""
    In this script I load and process Vitor amadeus-dataset-v6.json in order to compute the syntax similarity score bnetween each utterance and it list of generated paraphrases.
    In this script, I load and process the Vitor amadeus-dataset-v6.json file to calculate the syntactic similarity score between each statement and its list of generated paraphrases.
    The aim is to find out which is the best prompt to guide GPT to diverse syntax generation.

    Description of the amadeus-dataset-v6.json file created by Vitor:
        Jorge decided to gather 10 different APIs from the Amadeus API documentation, which are related to travel in general (booking flights, hotels, looking for recommendations of travels and so on).
        The documentation provides information such as description, base utterance, endpoints, parameters and so on. So he decided to pick 10 of them to use as the dataset of the paper.
        The JSON file with all the information is also added in this folder.

    To run the script please provide the path to amadeus-dataset-v6.json or any furhter version: e.g. python process_amadeus.py amadeus-dataset-v6.json
"""

def save_data_as_json(data):
    """
    Converts data to a JSON string and saves it to a file with a dynamic filename based on the current time.

    :args
        data: The data to be converted to JSON.

    :returns
        None

    :example
        >>> my_data = {'name': 'John', 'age': 25}
        >>> save_data_as_json(my_data)
    """
    # Get the current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert data to JSON string
    json_data = json.dumps(data)

    # Create the filename with the current time
    filename = f"processed-amadeus-dataset-v6-{current_time}.json"

    # Write JSON string to a file
    with open(filename, "w") as file:
        file.write(json_data)

def load_json_file(file_path):
    """
    Loads and returns the JSON data from the specified file path.

    :args
        file_path (str): The path to the JSON file.

    :returns
        dict: The loaded JSON data.

    :raises
        FileNotFoundError: If the specified file path does not exist.
        JSONDecodeError: If the JSON file is not valid and cannot be decoded.

    :Example
        file_path = 'path_to_your_file.json'
        json_data = load_json_file(file_path)
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_constituency_parse_tree(nlp,utterance):
    """
    Get the constituency parse tree of the given utterance using the Stanza library.

    :args
        nlp (stanza.pipeline.core.Pipeline): A Stanza Pipeline object for linguistic annotation.
        utterance (str): The input utterance to parse.

    :returns
        tree (stanza.models.constituency.parse_tree.Tree): The constituency parse tree of the utterance.
        root (str): The root label of the parse tree (should be 'ROOT').
        tree_without_root (Tree): The parse tree without the root label.

    :raises
        AssertionError: If the root label of the tree is not 'ROOT'.
    """

    doc = nlp(utterance)
    #print(f"doc.sentences: {doc.sentences[0].constituency}")
    tree = doc.sentences[0].constituency

    root = tree.label
    assert root == "ROOT", "label should be 'ROOT'"

    tree_without_root = tree.children[0]# remove the ROOT label form the tree e.g. (ROOT (S (VP (VB)))) => (S (VP (VB)))

    return tree,root,tree_without_root

def extract_pairwise_combinations(dictionary):
    """
    Extracts pairwise combinations of elements from a dictionary. Given a dictionary with string keys and values, this function generates
    and returns a list of tuples representing pairwise combinations of elements in the format of (key1, key2, value1, value2).

    :args
        dictionary (dict): A dictionary with string keys and values.

    :returns
        list: A list of tuples representing pairwise combinations of elements.

    :examples
        >>> my_dictionary = {
        ...     "key1": "value1",
        ...     "key2": "value2",
        ...     "key3": "value3"
        ... }
        >>> result = extract_pairwise_combinations(my_dictionary)
        >>> print(result)
        [('key1', 'key2', 'value1', 'value2'), ('key1', 'key3', 'value1', 'value3'), ('key2', 'key3', 'value2', 'value3')]
    """

    combinations = []
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1 = keys[i]
            key2 = keys[j]
            value1 = values[i]
            value2 = values[j]
            combinations.append((key1, key2, value1, value2))

    return combinations

def test_stanza():
    # set tokenize_no_ssplit to True to disable sentence segmentation, otherwise stanza will split the sentence every time it finds a dot (.)
    # e.g. if tokenize_no_ssplit=False. a = "I love monkeys. Birds too." stanza will split it into two sentences: s1 = "I love monkeys" and s2 = "Birds too."
    # However, if tokenize_no_ssplit=True, the stanza will split when it finds a line break and not a period,
    # e.g. a = "I love monkeys. Birds too." stanza will split it into one sentences: s = "I love monkeys. Birds too."
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', tokenize_no_ssplit=True)
    
    utterances = [
        'Book a flight from Lyon to Sydney. Tell me the weather in Bagneux, Paris for tomorrow.',
        "Tell me the weather in Bagneux, Paris for tomorrow.",
        "Tell me the weather in Bagneux, Paris for tomorrow",
        # "Can you provide the weather forecast for tomorrow in Bagneux, Paris?",
        # "I would like to know the weather in Bagneux, Paris for tomorrow, please.",
        # "Please tell me what the weather will be like in Bagneux, Paris tomorrow.",
        # "Do you have any information on the weather in Bagneux, Paris tomorrow?",
        # "Could you inform me about the weather in Bagneux, Paris for tomorrow?",
    ]

    syn_trees = dict()
    for utr in utterances:
        print(utr)
        _,_,tree_without_root = get_constituency_parse_tree(nlp,utr)
        syn_trees[utr] = str(tree_without_root)
    
    data = extract_pairwise_combinations(syn_trees)
    for combination in data:
        utr1, utr2, syn_tree1, syn_tree2 = combination
        a_tree = evt.get_parse_template(syn_tree1,2)
        b_tree = evt.get_parse_template(syn_tree2,2)
        ted = evt.compute_tree_edit_distance(a_tree,b_tree)
        print(f"ted('{utr1}','{utr2}'): {ted}")

def extract_values_by_key(dictionary_list, key):
    """
    Extracts values from a list of dictionaries based on a predefined key.
    Given a list of dictionaries and a key, this function iterates over each
    dictionary and extracts the corresponding value for the provided key.

    :args
        dictionary_list (list): A list of dictionaries.
        key (str): The key to extract the values from.

    :returns
        list: A list of values corresponding to the specified key.

    :examples
        >>> my_list = [
        ...     {"name": "John", "age": 25},
        ...     {"name": "Jane", "age": 30},
        ...     {"name": "Alice", "age": 35}
        ... ]
        >>> key = "name"
        >>> result = extract_values_by_key(my_list, key)
        >>> print(result)
        ['John', 'Jane', 'Alice']
    """
    values = []

    for dictionary in dictionary_list:
        if key in dictionary:
            values.append(dictionary[key])

    return values

def compute_ted_average_score(ted_scores):
    """
    Computes the average score from a list of scores.

    :args
        ted_scores (list): A list of Tree Edit distance scores.

    :returns
        float: The average score
    """
    if not ted_scores:
        raise "The ted_scores list is empty"

    total = sum(ted_scores)
    average = total / len(ted_scores)
    return average

def main():
    parser = argparse.ArgumentParser(description='Load JSON file')
    parser.add_argument('-f','--file_path', type=str, help='Path to the JSON file',required=True)
    args = parser.parse_args()

    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', tokenize_no_ssplit=True)
    
    json_data = load_json_file(args.file_path)

    result = dict()
    for k,v in json_data.items():
        generated_utt = v['generated_utt']
        generated_utt_dict = dict()
        for generated_utt_index, field in enumerate(generated_utt):
            utterances = field['utterances']
            syn_trees = dict()

            for utr in utterances:
                _,_,tree_without_root = get_constituency_parse_tree(nlp,utr)
                syn_trees[utr] = str(tree_without_root)

            data = extract_pairwise_combinations(syn_trees)

            ted_scores = dict()
            new_syntaxes = set()
            current_prompt = field['prompt']
            gpt_model = field['model']
            pairwise_mean_ted_score = []#this variable will contain the average mean TED score of the current list of utterances, generated using current_prompt
            for idx,combination in enumerate(data):
                utr1, utr2, syn_tree1, syn_tree2 = combination
                a_tree = evt.get_parse_template(syn_tree1,2)
                new_syntaxes.add(a_tree)

                b_tree = evt.get_parse_template(syn_tree2,2)
                new_syntaxes.add(b_tree)
                ted = evt.compute_tree_edit_distance(a_tree,b_tree)
                pairwise_mean_ted_score.append(ted)

                ted_scores[idx] = {
                    'u1': utr1,
                    'u2': utr2,
                    'ted':ted,
                    'syn_tree_u1':syn_tree1,
                    'syn_tree_u2':syn_tree2
                }
            ted_mean_score = compute_ted_average_score(pairwise_mean_ted_score)
            generated_utt_dict[generated_utt_index] = {
                'prompt': current_prompt,
                'model': gpt_model,
                'New_Syntax_templates_len': len(new_syntaxes),
                'New_Syntax_templates': list(new_syntaxes),#convert to list to avoid TypeError: Object of type set is not JSON serializable
                'Mean_edit_distance': ted_mean_score,
                'Tree_edit_distance': ted_scores
            }
        result[k] = generated_utt_dict
    
    save_data_as_json(result)

if __name__ == "__main__":
    main()