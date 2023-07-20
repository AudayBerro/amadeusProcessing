import argparse
import json

"""
    This script renders and prettifies the joson file generated by the process_amadeus.py script
    For reach API endpoint (first key in the JSON ile e.g. "amadeus.com:amadeus-flight-offers-search") the script only extract for each prompt utterance's the following values:
        1. New_Syntax_templates which is the list of new syntax tenplated extraced from the generated utterances
        2. Mean_edit_distance which is the average pairwise Tree Edit Distance of the generated utterances.
"""

def get_required_data(field):
    """
    Extracts required data from the given dictionary.
    
    :args
        data_field (dict): A dictionary with the following keys:
            - prompt: The GPT prompt that was used to generate the utterances.
            - model: The GPT model used for the generation (e.g., "gpt-3.5-turbo").
            - New_Syntax_templates_len: The length of the New_Syntax_templates set.
            - New_Syntax_templates: A list of new syntax templates extracted from the generated_utt.
            - Mean_edit_distance: The average pairwise Tree Edit Distance of the generated utterances in generated_utt list.
            - Tree_edit_distance: The pairwise Tree Edit Distance of the generated utterances in generated_utt list.

    :returns
        tuple: A tuple containing the retrieved data in the following order:
            - prompt (str): The GPT prompt.
            - model (str): The GPT model.
            - New_Syntax_templates_len (int): The length of the New_Syntax_templates set.
            - New_Syntax_templates (list): The list of new syntax templates.
            - Mean_edit_distance (float): The average pairwise Tree Edit Distance.
            - Tree_edit_distance (List): The pairwise Tree Edit Distance. Is a list of tuples(u1,u2, TED, syn_u1,syn_u2).
                - u1 (str): utterance 1
                - u2 (str): utterance 2
                - TED (float): Tree Edit distance between the syntactic tree template of u1 and u2 (respectively syn_u1 and syn_u1).
                - syn_u1 (str): constituenct parse tree template of the utterance 1.
                - syn_u2 (str): constituenct parse tree template of the utterance 2.
    """
    
    prompt = field['prompt']
    model = field['model']
    New_Syntax_templates_len = field['New_Syntax_templates_len']
    New_Syntax_templates = field['New_Syntax_templates']
    Mean_edit_distance = field['Mean_edit_distance']
    Tree_edit_distance = field['Tree_edit_distance']

    return  prompt, model, New_Syntax_templates_len, New_Syntax_templates,Mean_edit_distance,Tree_edit_distance

def prettify_data(file_path):

    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    result = dict()
    for k,field in json_data.items():
        curent_prompt_batch = dict()
        for idx,element in field.items():
            prompt, model, New_Syntax_templates_len, New_Syntax_templates, Mean_edit_distance, _ = get_required_data(element)
            curent_prompt_batch[idx] = {
                "prompt": prompt,
                "model": model,
                "New_Syntax_templates_len": New_Syntax_templates_len,
                "Mean_edit_distance": Mean_edit_distance,
                "New_Syntax_templates": New_Syntax_templates
            }
        result[k] = curent_prompt_batch
    
    # Convert data to JSON string
    json_data = json.dumps(result)

    # Create the filename with the current time
    filename = f"pretified-{file_path}"

    # Write JSON string to a file
    with open(filename, "w") as file:
        file.write(json_data)

def generate_summary(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    result = {
        '0': {'New_Syntax_templates_len': [], 'Mean_edit_distance': [], 'Syntax_templates_set': set(), 'len_Syntax_templates_set': 0},
        '1': {'New_Syntax_templates_len': [], 'Mean_edit_distance': [], 'Syntax_templates_set': set(), 'len_Syntax_templates_set': 0},
        '2': {'New_Syntax_templates_len': [], 'Mean_edit_distance': [], 'Syntax_templates_set': set(), 'len_Syntax_templates_set': 0},
        '3': {'New_Syntax_templates_len': [], 'Mean_edit_distance': [], 'Syntax_templates_set': set(), 'len_Syntax_templates_set': 0},
        '4': {'New_Syntax_templates_len': [], 'Mean_edit_distance': [], 'Syntax_templates_set': set(), 'len_Syntax_templates_set': 0}
    }
    
    for k,field in json_data.items():
        for idx,element in field.items():
            New_Syntax_templates_len = element['New_Syntax_templates_len']
            Mean_edit_distance = element['Mean_edit_distance']
            Syntax_templates_set = element['New_Syntax_templates']#get list of unique syntax templates 

            result[idx]['New_Syntax_templates_len'].append(New_Syntax_templates_len)
            result[idx]['Mean_edit_distance'].append(Mean_edit_distance)
            result[idx]['Syntax_templates_set'].update(Syntax_templates_set)
    
    #add length of Syntax_templates_set
    for k in result.keys():
        l = len(result[k]['Syntax_templates_set'])
        result[k]['len_Syntax_templates_set'] = l
        result[k]['Syntax_templates_set'] = list( result[k]['Syntax_templates_set'] )
    
    # Create the filename with the current time
    filename = f"corpus-summary.json"

    # Convert data to JSON string
    json_data = json.dumps(result)

    # Write JSON string to a file
    with open(filename, "w") as file:
        file.write(json_data)
    
    return result

def plot_corpus_summary_bar(data):
    import matplotlib.pyplot as plt

    # Extract the desired value from the nested dictionaries
    X = ['0', '1', '2', '3', '4']
    y_values = [nested_dict['len_Syntax_templates_set'] for nested_dict in data.values()]

    print(X)
    print(y_values)

    plt.bar(X,y_values)
    plt.ylabel('Number of New Syntax templates')
    plt.xlabel('Prompt')
    plt.title('Syntax templates per prompt')

    # Display the bar plot
    plt.show()

def plot_bar_per_intent(raw_data):
    import matplotlib.pyplot as plt
    import numpy as np

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

    # Extract the desired value from the nested dictionaries
    labels = [x_label.split(":")[1] for x_label in raw_data.keys()]
    labels = labels[5:11]
    print(f"labels: {labels}")

    data = []
    for prompt in raw_data.values():
        current_intent = []#For each intent group the New_Syntax_templates_len value per prompt.
        for nested_key,nested_value in prompt.items():
            #print(nested_key)#print 0 1 2 3 4
            current_intent.append( nested_value['New_Syntax_templates_len'])
        data.append(current_intent)
    
    X = np.arange(len(labels))
    print(f"data: {data[0]}")
    print(X)
    width = 0.25

    fig, ax = plt.subplots()

    rects0 = ax.bar(X - 1/2*width, data[5], color='r', width=width/2)
    rects1 = ax.bar(X - width, data[6], color='g', width=width/2)
    rects2 = ax.bar(X, data[7], color='blue', width=width/2)
    rects3 = ax.bar(X + width, data[8], color='orange', width=width/2)
    rects4 = ax.bar(X + 1/2*width, data[9], color='skyblue', width=width/2)

    #Syntax Template Distribution by Intent, Organized by Prompt
    ax.set_ylabel('Number of New Syntax templates')
    ax.set_xlabel('Amadeus Intent')
    ax.set_title('Analyzing Prompt-Grouped Syntax Templates for Different Intents')
    ax.set_xticks(X)
    ax.set_xticklabels(labels)
    ax.legend([0,1,2,3,4], title="Prompt")
    #plt.xlabel('Amadeus API intent')
    autolabel(rects0)
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load JSON file')
    parser.add_argument('-f','--file_path', type=str, help='Path to the JSON file',required=True)
    args = parser.parse_args()

    # prettify_data(args.file_path)

    # a = generate_summary(args.file_path)
    # plot_his_corpus_summary(a)

    with open(args.file_path, 'r') as file:
        json_data = json.load(file)
    
    plot_bar_per_intent(json_data)
    