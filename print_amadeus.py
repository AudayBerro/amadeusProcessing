import argparse
import json

"""
    This script is an UtilityToola to read and process the amadeus-dataset-v6.json in order to print varuious intersting selected fields. For example print only the prompt or the generated utternaces, etc.
"""
def save_data_as_json(data,filename):
    """
    Converts data to a JSON string and saves it to a file with a dynamic filename based on the current time.

    :args
        data: The data to be converted to JSON.
        filename: the name of the file

    :returns
        None

    :example
        >>> my_data = {'name': 'John', 'age': 25}
        >>> save_data_as_json(my_data)
    """

    # Convert data to JSON string
    json_data = json.dumps(data)

    # Write JSON string to a file
    with open(filename, "w") as file:
        file.write(json_data)

def display_selected_intent_paraphrases(json_data, selected_intent):
    """
    Print the paraphrase set of a selected intent from the amadeus-dataset-v6.json.

    :args
        json_data (dict): A dictionary containing the data extracted from the amadeus-dataset-v6.json.
        selected_intent (str): The intent for which paraphrases need to be displayed.

    :returns
        None: The function doesn't return any value. It prints the paraphrases directly.

    :description
        This function takes the json_data dictionary and a selected_intent as input and prints the paraphrases associated with the specified intent.
        The paraphrases are extracted from the 'generated_utt' field of the given json_data. The function first extracts all the utterances related to the selected_intent,
        including 'base_utt' and 'test_utt', and then prints the generated paraphrases for that intent.

        Example Usage:
            json_data = {
                'intent1': {
                    'base_utt': 'Base utterance for intent1',
                    'test_utt': ['Test utterance 1', 'Test utterance 2'],
                    'generated_utt': [
                        {'utterances': ['Generated paraphrase 1', 'Generated paraphrase 2']},
                        {'utterances': ['Generated paraphrase 3', 'Generated paraphrase 4']}
                    ]
                },
                'intent2': {
                    # Data for intent2
                },
                # Other intents
            }

            >>> display_paraphrases_only(json_data, 'intent1')
            # Output will be the generated paraphrases for 'intent1'.
    """
    result = dict()

    utterances = []  # add base_utt and test_utt item
    paraphrases = []  # add generated_utt
    result['metadata'] = "This file contains the generated paraphrases for each intent extracted from the amadeus-dataset-v6.json"
    for k, v in json_data.items():
        base_utt = v['base_utt']
        utterances.append(base_utt)

        test_utt = v['test_utt']
        utterances.extend(test_utt)

        utterances = list(map(str.strip, utterances))  # removes any leading and trailing whitespaces

        generated_utt = v['generated_utt']
        current_prompt = dict()
        for generated_utt_index, field in enumerate(generated_utt):
            if k == selected_intent:
                for paraph in field['utterances']:
                    print(f"prompt-{generated_utt_index}: {paraph}")

def display_paraphrases_only(json_data):
    """
    Extract paraphrases from the 'amadeus-dataset-v6.json' and save them in a new JSON file.

    :args
        json_data (dict): A dictionary containing the data extracted from the 'amadeus-dataset-v6.json'.

    :returns
        None: The function doesn't return any value. It saves the extracted paraphrases in a new JSON file.

    :description
        This function takes the 'json_data' dictionary as input, which contains the paraphrase data for each intent
        extracted from the 'amadeus-dataset-v6.json' file. It processes the data, extracting the paraphrases for each intent,
        and saves them in a new JSON file named "paraphrases-amadeus-dataset-v6.json".

        The function iterates through each intent in 'json_data', extracts 'base_utt' and 'test_utt' items as 'utterances',
        and 'generated_utt' as 'paraphrases'. It then organizes this data into a dictionary format and saves it in the 'result' dictionary.

        The generated dictionary format for 'result' will be like this:
        {
            'intent1': {
                0: {
                    'seed_utterances': ['Base utterance for intent1', 'Test utterance 1', 'Test utterance 2'],
                    'len_seed_utterances': 3,
                    'paraphrases': ['Generated paraphrase 1', 'Generated paraphrase 2', ...],
                    'len_paraphrases': <length_of_generated_paraphrases_list>,
                },
                1: {
                    # Data for the next set of paraphrases (if available)
                },
            },
            'intent2': {
                # Data for the next intent (if available)
            },
            ...
        }

        After processing all the intents, the 'result' dictionary is converted to a JSON string and saved in a new file named "paraphrases-amadeus-dataset-v6sdsd.json".

    :Example Usage
        >>> json_data = {
            'intent1': {
                'base_utt': 'Base utterance for intent1',
                'test_utt': ['Test utterance 1', 'Test utterance 2'],
                'generated_utt': [
                    {'utterances': ['Generated paraphrase 1', 'Generated paraphrase 2']},
                    {'utterances': ['Generated paraphrase 3', 'Generated paraphrase 4']}
                ]
            },
            'intent2': {
                # Data for intent2
            },
            # Other intents
        }

        >>> display_paraphrases_only(json_data)
        # Paraphrase data will be saved in "paraphrases-amadeus-dataset-v6sdsd.json" file.
    """
    result = dict()

    utterances_list = []  # List to store all utterances for each intent
    paraphrases_list = []  # List to store all paraphrases for each intent
    result['metadata'] = "This file contains the generated paraphrases for each intent extracted from the amadeus-dataset-v6.json"
    
    for intent, intent_data in json_data.items():
        base_utt = intent_data['base_utt']
        utterances_list.append(base_utt)

        test_utt = intent_data['test_utt']
        utterances_list.extend(test_utt)

        utterances_list = list(map(str.strip, utterances_list))  # Remove leading and trailing whitespaces from the utterances
        
        generated_utt = intent_data['generated_utt']
        current_prompt = dict()
        for generated_utt_index, field in enumerate(generated_utt):
            paraphrases_list.extend(field['utterances'])

        current_prompt[generated_utt_index] = {
            'seed_utterances': utterances_list.copy(),
            'len_seed_utterances': len(utterances_list),
            'paraphrases': paraphrases_list.copy(),
            'len_paraphrases': len(paraphrases_list),
        }
        result[intent] = current_prompt
        utterances_list.clear()
        paraphrases_list.clear()

    # Create the filename with the current time
    filename = f"paraphrases-amadeus-dataset-v6.json"

    # Write JSON string to a file
    save_data_as_json(result,filename)

def display_intent_parameter(json_data, selected_intent):
    """
    Display the parameters of the selected intent. This function takes in a JSON data dictionary and the name of a selected intent. It then retrieves the parameters of the selected intent from
    the JSON data and prints information about each parameter, including its name, type, whether it is required or not, mentions, and values.

    :args
        json_data (dict): A dictionary containing intent data in JSON format.
        selected_intent (str): The name of the selected intent.

    :returns
        None

    :Example
        >>> json_data = {
            "intent1": {
                "paths": {
                    "endpoint1": {
                        "parameters": [
                            {
                                "name": "param1",
                                "type": "string",
                                "required": True,
                                "mentions": 2,
                                "values": ["value1", "value2"]
                            },
                            {
                                "name": "param2",
                                "type": "integer",
                                "required": False,
                                "mentions": 5,
                                "values": [1, 2, 3]
                            }
                        ]
                    }
                }
            }
        }

        >>> display_intent_parameter(json_data, "intent1")

        Output:
            1. 'param1' - string - True
            mentions: 2
            values: ['value1', 'value2']

            2. 'param2' - integer - False
            mentions: 5
            values: [1, 2, 3]

            The intent has 2 parameters with 1 required.
            List of required parameters: ['param1 - string']
    """

    intent_data = json_data[selected_intent]
    paths = intent_data['paths']# get list of parameters

    for endpoint,endpoint_data in paths.items():
        parameters = endpoint_data['parameters']
        number_of_parameters = len(parameters)

        required_param_counter = 0
        required_params = []
        for idx, param in enumerate(parameters):
            if param['required']:
                required_param_counter+=1
                required_params.append(f"{ param['name']} - {param['type']}")
            print(f"{idx+1}. '{param['name']}' - {param['type']} - {param['required']}")
            print(f"mentions: {param['mentions']}")
            print(f"values: {param['values']}")
            print()
        
        print(f"The intent has {number_of_parameters} parameters with {required_param_counter} required.")
        print(f"List of required parameters: {required_params}")

def display_all_intent_parameter(json_data):
    """
    This function is a wrapper to display the parameters of all the intents in the amadeus dataset.
    It takes in a JSON data dictionary containing intent data and prints information about each parameter,
    including its name, type, whether it is required or not, mentions, and values.

    :args
        json_data: A json object containing the amadeus dataset.

    :return: None
    """
    for intent, intent_data in json_data.items():
        print()
        print(intent)
        display_intent_parameter(json_data,intent)
        print()

def get_selected_metric(json_data, metric = 0):
    """
    Retrieve the selected metric used for outlier detection from the amadeus dataset.

    This function takes a JSON object representing the amadeus dataset and an integer metric code (0, 1, or 2),
    which corresponds to the specific metric used to detect outliers in the dataset.

    :args
        json_data: A dictionary containing the amadeus dataset in JSON format (Dict[str, Any]).
        metric: The code representing the metric used for outlier detection.
            0: Precision Score.
            1: Recall Score.
            2: F1 Score.
            The default value is 0.

    :return
        The name of the selected metric used for outlier detection (str).

    :raises
        KeyError: If the provided metric code is not valid for the given dataset.

    :examples
        >>> json_data = {...}  # Replace '...' with the actual JSON object containing the dataset.
        >>> selected_metric = get_selected_metric(json_data, metric=1)
        >>> print(selected_metric)
        "Recall Score"

    :Note
        The function assumes that the provided JSON object contains a valid amadeus dataset with appropriate keys and nested structures.
        It extracts the selected metric name from the dataset's outlier scores section for further analysis or reporting purposes.
    """
    json_data_keys = json_data.keys()
    json_data_keys = list(json_data_keys)#get a hand on the intent
    frst_intnt_idx = json_data_keys[0]#just select the first intent
    prompts_ = json_data[frst_intnt_idx]#get the prompts and their nested data => {'0':{...}, '1':{...}, '2':{...}, '3':{...}, '4':{...}}
    prmpt_keys = prompts_.keys()#get the key to be able to iterate => dict_keys(['0', '1', '2', '3', '4'])
    prmpt_keys = list(prmpt_keys)
    prmpt_0_keys = prmpt_keys[0]#just select the first prompt
    outliers_scores_items_keys = json_data[frst_intnt_idx][prmpt_0_keys]['outlier scores'].keys()
    outliers_scores_items_keys = list(outliers_scores_items_keys)
    outliers_scores_items_0 = outliers_scores_items_keys[0]#just select the first generated utterance/paraphrase
    # print(json_data[frst_intnt_idx][prmpt_0_keys]['seed_utterances'])
    outliers_scores_items_0_scores = json_data[frst_intnt_idx][prmpt_0_keys]['outlier scores'][outliers_scores_items_0]#get the precision, recall and F1 scores of the outliers_scores_items_0
    outliers_metrics_keys = outliers_scores_items_0_scores.keys()
    outliers_metrics_keys = list(outliers_metrics_keys)

    selected_metric_score = outliers_metrics_keys[metric]# which metric was used to detect outliers

    return selected_metric_score

def display_outlier_bert_score_output(json_data, metric = 0):
    """
    This function si a wrapper to display the parameters of all the intents in the amadeus dataset. This function takes in a JSON data dictionary.
    It then retrieves the parameters of all intent from the JSON data and prints information about each parameter, including its name, type, whether it is required or not, mentions, and values.

    metric: Which metric was used to detect the outlier:
        0: Precision_Score
        1: Recall_Score
        2: F1_Score.

    """

    # avoid the 'metadata' item
    json_data = {key: value for key, value in json_data.items() if key != 'metadata'}

    json_data_keys = json_data.keys()
    json_data_keys = list(json_data_keys)#get a hand on the intent
    frst_intnt_idx = json_data_keys[0]#just select the first intent
    prompts_ = json_data[frst_intnt_idx]#get the prompts and their nested data => {'0':{...}, '1':{...}, '2':{...}, '3':{...}, '4':{...}}
    prmpt_keys = prompts_.keys()#get the key to be able to iterate => dict_keys(['0', '1', '2', '3', '4'])
    prmpt_keys = list(prmpt_keys)
    prmpt_0_keys = prmpt_keys[0]#just select the first prompt
    outliers_scores_items_keys = json_data[frst_intnt_idx][prmpt_0_keys]['outlier scores'].keys()
    outliers_scores_items_keys = list(outliers_scores_items_keys)
    outliers_scores_items_0 = outliers_scores_items_keys[0]#just select the first generated utterance/paraphrase
    # print(json_data[frst_intnt_idx][prmpt_0_keys]['seed_utterances'])
    outliers_scores_items_0_scores = json_data[frst_intnt_idx][prmpt_0_keys]['outlier scores'][outliers_scores_items_0]#get the precision, recall and F1 scores of the outliers_scores_items_0
    outliers_metrics_keys = outliers_scores_items_0_scores.keys()
    outliers_metrics_keys = list(outliers_metrics_keys)

    selected_metric_score = outliers_metrics_keys[metric]# which metric was used to detect outliers

    sys.exit()
    supported_metrics = 2
    supported_metrics = 2
    selected_metric = supported_metrics[metric]
    selected_metric_score = paraphrase_scores[selected_metric]

    for intent, intent_data in json_data.items():
        print(intent)

        prompt_key = intent_data.keys()#a variable used to extract utterances once and for all, since all prompt have the same seed utterances list
        prompt_key = list(prompt_key)# convert to avoid AttributeError: 'dict_keys' object has no attribute 'list'
        frst_prmpt_idx = prompt_key[0]#just select the first prompt

        utterances = intent_data[frst_prmpt_idx]['seed_utterances']
        print()
        print("Seed utterances:")
        print("\n".join([f"\t{idx+1}. {utr}" for idx, utr in enumerate(utterances)]))#print seed utterances
        print()
        
        for prompt, data in intent_data.items():
            outlier_paraphrases = data['outlier_paraphrases']#get lsit of outliers paraphrases
            paraphrases = data['outlier scores']#get lsit of all generated paraphrases despite the outliers detection
            for paraphrase, paraphrase_scores in paraphrases.items():
                if paraphrase in outlier_paraphrases:
                    print(f"{paraphrase} - outlier - {Precision_Score}: {selected_metric_score}")
                else:
                    print(f"{paraphrase} - good - {Precision_Score}: {selected_metric_score}")
            sys.exit()

        current_prompt[generated_utt_index] = {
            'seed_utterances': utterances_list.copy(),
            'len_seed_utterances': len(utterances_list),
            'paraphrases': paraphrases_list.copy(),
            'len_paraphrases': len(paraphrases_list),
        }
        result[intent] = current_prompt
        utterances_list.clear()
        paraphrases_list.clear()

    # # Create the filename with the current time
    # filename = f"paraphrases-amadeus-dataset-v6.json"

    # # Write JSON string to a file
    # save_data_as_json(result,filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load JSON file')
    parser.add_argument('-f','--file_path', type=str, help='Path to the JSON file',required=True)
    args = parser.parse_args()

    with open(args.file_path, 'r') as file:
        json_data = json.load(file)
    
    # selected_intent = "amadeus.com:amadeus-location-score"
    # display_selected_intent_paraphrases(json_data,selected_intent)
    # display_paraphrases_only(json_data)
    # display_all_intent_parameter(json_data)
    # display_intent_parameter(json_data,selected_intent)
    display_outlier_bert_score_output(json_data)