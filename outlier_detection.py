import argparse
import json
import logging

import tensorflow_hub as hub
import tensorflow as tf

from bert_score import BERTScorer
import transformers

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection


# hide the loading messages
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

"""
    In this script i impelemnted a paraprhases outlier detection system, I just replicate the approach taken by Outlier Detection for Improved Data Quality and Diversity in Dialog Systems
    paper: https://aclanthology.org/N19-1051.pdf

    Main idea compute mean vector of cluster of paraprhases and compare each candidate to the mean, if close related otherwise it is an outlier. We ahve two types of outlier according to the paper errors and unique. Errors, sentences that
    have been mislabeled whose inclusion in the dataset would be detrimental to model performance. Unique, sentences that differ in structure or content from most in the data and whose inclusion would be helpful for model robustness.
    
    paper appproach to detect outliers in a dataset as follows:
        1. Generate a vector representation of each instance.
        2. Average vectors to get a mean representation => they used USE, ELMO, Glove, combination of them => I will LLM like use GPT and USE.
        3. Calculate the distance of each instance from the mean.
        4. Rank by distance in ascending order.
        5. (Cut off the list, keeping only the top k% as outliers.)
"""

def compute_cosine_similarity(tensor1, tensor2):
    """
    Computes the cosine similarity between two TensorFlow tensors.

    :args
        tensor1: A TensorFlow tensor.
        tensor2: A TensorFlow tensor with the same shape as tensor1.

    :return
        A TensorFlow scalar representing the cosine similarity between the two tensors.
    """
    # Ensure tensors have the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("tensor1 and tensor2 must have the same shape.")
    
    # Convert TensorFlow vectors to NumPy arrays
    vector1_np = tensor1.numpy()
    vector2_np = tensor2.numpy()

    # Reshape vectors if necessary
    # Reshape vectors to 2D arrays
    vector1_np = np.expand_dims(vector1_np, axis=0)
    vector2_np = np.expand_dims(vector2_np, axis=0)

    # Compute cosine similarity using sklearn.metrics.pairwise.cosine_similarity
    # similarity is a numpy.float32
    similarity = cosine_similarity(vector1_np, vector2_np)[0][0]

    return similarity.item()

def save_data_as_json(data,file_name):
    """
    Converts data to a JSON string and saves it to a file with a dynamic filename based on the current time.

    :args
        data: A pyhton dictionary to be converted to JSON.
        file_name: name to assign to the stroed file e.g. outlier-detection-amadeus-dataset-v6-bert-score

    :returns
        None

    :example
        >>> my_data = {'name': 'John', 'age': 25}
        >>> save_data_as_json(my_data)
    """
    from datetime import datetime
    # Get the current time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert data to JSON string
    json_data = json.dumps(data)

    # Create the filename with the current time
    filename = f"{file_name}-{current_time}.json"

    # Write JSON string to a file
    with open(filename, "w") as file:
        file.write(json_data)

def compute_euclidean_distance(vector1, vector2):  
    """
    Calculates the Euclidean distance between two TensorFlow vectors.

    :args
        vector1: A TensorFlow vector.
        vector2: A TensorFlow vector with the same shape as vector1.

    :return
        A TensorFlow scalar representing the Euclidean distance between the two vectors.

    :raises
        ValueError: If the shapes of vector1 and vector2 do not match.
    """
    
    # return pairwise euclidead difference matrix
    # euclidean_distance is a tensorflow.python.framework.ops.EagerTensor extract the value to avoid TypeError: Object of type EagerTensor is not JSON serializable
    euclidean_distance = tf.norm(vector1 - vector2, ord='euclidean')

    return euclidean_distance.numpy().item()

def get_mean_average_vector(embeddings_list):
    """
    Calculates the mean average vector of a list of TensorFlow embeddings.

    :args
        embeddings_list: A list of TensorFlow embeddings.

    :return
        A TensorFlow tensor representing the mean average vector.

    :raises
        ValueError: If the embeddings_list is empty or if the embeddings have inconsistent shapes.
    """

    if len(embeddings_list) == 0:
        raise ValueError("Empty embeddings_list. Cannot calculate the mean average vector.")

    # Convert embeddings_list to a NumPy array
    embeddings_array = np.array(embeddings_list)

    # Check if all embeddings have the same shape
    if not np.all(embeddings_array.shape[1:] == embeddings_array[0].shape):
        raise ValueError("Inconsistent shapes in the embeddings_list.")

    # Calculate the mean average vector using np.mean()
    mean_vector = np.mean(embeddings_array, axis=0)

    # Convert mean_vector back to a TensorFlow tensor
    mean_tensor = tf.convert_to_tensor(mean_vector)

    return mean_tensor

def print_scores(utterance, euclidean_distance, cosine_similarity):
    """
    Prints the scores (euclidean distance and cosine similarity) for an utterance.

    :args
        utterance: The input sentence.
        euclidean_distance: The euclidean distance score.
        cosine_similarity: The cosine similarity score.

    :returns
        None

    :prints
        The utterance, euclidean distance score, and cosine similarity score.
    """
    print(
            f"\nSentence: {utterance}"\
            f"\n\t- euclidean distance: {euclidean_distance}"\
            f"\n\t- cosine similarity: {cosine_similarity}"
        )

def sort_dict_by_value(dictionary,selected_element):
    """
    Sorts a dictionary based on the values of selected_element.

    :args
        dictionary (dict): The dictionary to be sorted.
        selected_element (str): The key of the selected element to use for sorting.

    :returns
        dict: The sorted dictionary based on the values.
    
    Note:
        - If the values of the dictionary or nested dictionaries are not comparable (e.g., non-numeric values),
          the sorting order may not produce the expected result.
        - The selected_element parameter is used as a key to specify which element's value should be considered
          for sorting when encountering nested dictionaries.

    Example:
        >>> my_dict = {'apple': {'quantity': 5, 'price': 2}, 'banana': {'quantity': 2, 'price': 4}, 'orange': {'quantity': 8, 'price': 1}, 'grape': {'quantity': 3, 'price': 3}}
        >>> sorted_dict = sort_dict_by_value(my_dict, 'price')
        >>> print(sorted_dict)
        {'orange': {'quantity': 8, 'price': 1}, 'apple': {'quantity': 5, 'price': 2}, 'grape': {'quantity': 3, 'price': 3}, 'banana': {'quantity': 2, 'price': 4}}
    
    Note:
        - The sorting is performed based on the values of the selected_element.
        - The selected_element parameter should be a key that exists in the nested dictionaries of the input dictionary.
    """
    
    res = sorted(dictionary.items(), key = lambda x: x[1][selected_element])

    result_dict = {item[0]: item[1] for item in res}

    return result_dict

def get_scores(embed,utterances):
    """
    Computes scores for utterances based on embeddings.

    :args
        embed: an isntance of a Universal Sentence Encoder object,  type: tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject
        utterances: A list of utterances for testing.

    :prints
        - Average vectors computed from the embeddings.
        - Distances (Euclidean and cosine similarity) between the average vector and the embeddings of the provided utterances.
        - Distances (Euclidean and cosine similarity) between the average vector and the embeddings of a test sentence.
    """

    embeddings = embed(utterances)

    print(f" *Average vecotrs* ".center(30,"="))
    mean_tensor = get_mean_average_vector(embeddings)

    print("...done")

    print(f" *Compute distances* ".center(30,"="))

    json_dict_scores = dict()

    for i in range(len(utterances)):
        euclidean_score = compute_euclidean_distance(mean_tensor,embeddings[i])
        cosine_score = compute_cosine_similarity(mean_tensor,embeddings[i])
        #print_scores(utterances[i],euclidean_score,cosine_score)
        json_dict_scores[utterances[i]] = {
            'euclidean_distance': euclidean_score,
            'cosine_similarity': cosine_score
        }

    test_sentence = [
        "Book a flight from Lyon to Sydney"
    ]

    embeddings_test = embed(test_sentence)
    embeddings_list = [ emb for emb in embeddings]
    embeddings_list.append(embeddings_test[0])

    test_euclidean_score = compute_euclidean_distance(mean_tensor,embeddings_test[0])
    test_cosine_score = compute_cosine_similarity(mean_tensor,embeddings_test[0])
    #print_scores(test_sentence,test_euclidean_score,test_cosine_score)

    json_dict_scores[test_sentence[0]] = {
        'euclidean_distance': test_euclidean_score,
        'cosine_similarity': test_cosine_score
    }

    return mean_tensor, embeddings, json_dict_scores

def plot_LOF_outlier(X,X_scores,n_errors,n_neighbors,contamination):
    """
    Plots the results of local outlier factor (LOF) outlier detection.
    This function creates a scatter plot to visualize the results of local outlier factor (LOF) outlier detection.
    It plots the data points with circles whose radius is proportional to the outlier scores.

    :args
        X (array-like): The data points to be plotted.
        X_scores (array-like): The LOF scores associated with each data point.
        n_errors (List(float)): The number of errors in the predicted labels compared to the ground truth labels.
        n_neighbors (int): Value of the n_neighbors to use in the Plot title
        contamination ('auto’ or float, optional): Value of the contamination to use in the Plot title

    :returns
        None

    :Note
        This function requires the `matplotlib.pyplot` and `matplotlib.legend_handler.HandlerPathCollection` modules.
    """

    def update_legend_marker_size(handle, orig):
        "Customize size of the legend marker"
        handle.update_from(orig)
        handle.set_sizes([20])


    fig, ax = plt.subplots(figsize=(15, 10))  

    ax.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        s=1000 * radius,
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )
    
    # Add a legend
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax.legend(
        handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)},
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.35),
    )
    ax.axis("tight")
    ax.set_xlabel("prediction errors: %d" % (n_errors))
    print(f"\nn_neighbors = {n_neighbors} and contamination = {contamination} - X_scores: {X_scores}")

    ax.set_title(f"Local Outlier Factor (LOF) with n_neighbors = {n_neighbors} and contamination = {contamination}.")
    fig.savefig("./LOF_plots/best_parameter/plot_param_" + str(n_neighbors) + "_" + str(contamination) + ".png")
    #plt.show()
    plt.close()

def LOF_outlier(inliers_embeddings,outliers_embeddings,n_neighbors=20,contamination=0.1):
    """
    Performs local outlier factor (LOF) outlier detection on given inliers and outliers embeddings.
    Introduction to LOF https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection
    LOF documentation https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor

    This function applies the LOF algorithm to identify outliers in the given data.
    The LOF algorithm estimates the 'degree of abnormality' for each sample based on the ratio of its local reachability density and that of its k-nearest neighbors.
    Inliers tend to have a LOF score close to 1 (negative_outlier_factor_ or the X_scores in the return close to -1), while outliers tend to have a larger LOF score.
    The LOF scores provide a measure of the abnormality or outlierness of each sample.

    The function converts the `inliers_embeddings` and `outliers_embeddings` to NumPy arrays and concatenates them into a single array `X`.
    It then calculates the number of outliers, creates a ground truth array, and fits the LOF model to the data.
    The predicted labels and the number of errors in the predicted labels compared to the ground truth labels are determined.
    Finally, the LOF scores, predicted labels, and ground truth labels are returned as a tuple.

    :args
        inliers_embeddings (array-like): Embeddings of inliers.
        outliers_embeddings (array-like): Embeddings of outliers.
        n_neighbors (int, optional): Number of neighbors to use by default for kneighbors queries. If n_neighbors is larger than the number of samples provided, all samples will be used.
                                     The number of neighbors considered (parameter n_neighbors) is typically set 1) greater than the minimum number of samples a cluster has to contain, so that
                                     other samples can be local outliers relative to this cluster, and 2) smaller than the maximum number of close by samples that can potentially be local outliers.
                                     In practice, such information is generally not available, and taking n_neighbors=20 appears to work well in general.
        contamination ('auto’ or float, optional): Expected proportion of outliers in the data. Default is 0.1.
                                     The amount of contamination of the dataset, i.e. the proportion of outliers in the data set. When fitting this is used to define the threshold on the scores of the samples.
                                     - if ‘auto’, the threshold is determined as in the original paper,
                                     - if a float, the contamination should be in the range (0, 0.5].

    :returns
        tuple: A tuple containing the LOF scores, predicted labels, and ground truth labels.

    Note:
        This function requires the `numpy` and `sklearn.neighbors` modules.
    """

    X_inliers = inliers_embeddings.numpy()
    X_outliers = outliers_embeddings.numpy()

    X = np.r_[X_inliers, X_outliers]

    n_outliers = len(X_outliers)
    ground_truth = np.ones(len(X), dtype=int)
    ground_truth[-n_outliers:] = -1


    # Fit the model for outlier detection (default)
    from sklearn.neighbors import LocalOutlierFactor
    
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)#number_neighbors to consider and contamination expected proportion of outliers in the data
    y_pred = clf.fit_predict(X)
    n_errors = (y_pred != ground_truth).sum()#number of errors in the predicted labels compared to the ground truth labels

    X_scores = clf.negative_outlier_factor_

    plot_LOF_outlier(X,X_scores,n_errors,n_neighbors,contamination)

    return X_scores,y_pred,ground_truth

def print_xscores(X_scores,utterances,y_pred,ground_truth):
    """
    Prints the LOF scores, utterances, and identifies erroneous samples.
    This function generates and prints the LOF scores with their corresponding utterances in the format "utr: [utterance] - score: [LOF score]".
    It also identifies erroneous samples by comparing the predicted labels (`y_pred`) with the ground truth labels (`ground_truth`).

    The `X_scores` array contains the LOF scores for each data point, and the `utterances` list holds the corresponding utterances.
    The `y_pred` and `ground_truth` arrays represent the predicted and ground truth labels, respectively.

    The function prints the data with the label "Samples LOF predicted scores:", followed by the individual utterances and their LOF scores.
    It then identifies the erroneous samples and prints the number of prediction errors along with the list of samples that are concerned.
    Finally, it returns a tuple containing the list of erroneous samples and the list of printed data.

    :args
        X_scores (array-like): LOF scores for each data point.
        utterances (list): List of utterances corresponding to the LOF scores.
        y_pred (array-like): Predicted labels.
        ground_truth (array-like): Ground truth labels.

    :returns
        tuple: A tuple containing the list of erroneous samples and the list of printed data.

    Note:
        The `utterances`, `y_pred`, and `ground_truth` should have the same length.

    """

    data = [f"\t{indx+1}. {utr} - score: {scr}" for indx, (utr, scr) in enumerate(zip(utterances, X_scores))]

    
    print("\nPredicted LOF scores for the samples:")
    _ = [ print(d) for d in data ]
    
    erroneous_samples = [utterances[indx] for indx in range(len(utterances)) if y_pred[indx] != ground_truth[indx]]

    print(f"\nIt has {len(erroneous_samples)} prediction errors, here are the samples concerned:")
    _ = [ print(f"\t{indx+1}. {erroneous_samples[indx]}") for indx in range(len(erroneous_samples)) ]

    return erroneous_samples,data

def find_best_parameter(inliers_embeddings,outliers_embeddings,utterances):
    """
    Finds the best parameters for local outlier factor (LOF) outlier detection.
    This function searches for the best parameters for LOF outlier detection by iterating through different combinations of `n_neighbors` and `contamination` values. 
    It prints the parameter combinations, calculates the LOF scores, and identifies erroneous samples for each combination. It collects the configuration information in a list and returns it.

    The `inliers_embeddings` and `outliers_embeddings` are arrays containing the embeddings of inliers and outliers, respectively. The `utterances` list holds the corresponding utterances for the data points.

    :args
        inliers_embeddings (array-like): Embeddings of inliers.
        outliers_embeddings (array-like): Embeddings of outliers.
        utterances (list): List of utterances.

    :returns
        list: List of configurations with the best parameters.
    """
    import itertools

    a = range(1,21)
    b = [0.1,.2,0.3,0.4,0.5,'auto']
    config = []

    for n_neighbors,contamination in itertools.product(a,b):
        print(f"n_neighbors:{n_neighbors} contamination: {contamination}")
        X_scores,y_pred,ground_truth = LOF_outlier(inliers_embeddings,outliers_embeddings,n_neighbors,contamination)
        print_xscores(X_scores,utterances,y_pred,ground_truth)
        n_errors = (y_pred != ground_truth).sum()
        config.append({
            "n_neighbors": n_neighbors,
            "contamination": contamination,
            "n_errors": n_errors
        })

    best_config = []
    print(config)
    for cfg in config:
        if cfg["n_errors"]==0:
            best_config.append(cfg)
    return config,best_config

def test(embed):
    # paraphrases for which you want to create embeddings, passed as an array in embed()
    # the paraphrases are taken from the following paper: A Semantic Similarity Approach to Paraphrase Detection - Fernando, Samuel, and Mark Stevenson
    noisy_sentences = [
        "The Iraqi Foreign Minister warned of disastrous consequences if Turkey launched an invasion of Iraq",
        "Iraq has warned that a Turkish incursion would have disastrous results",
        "Iaq has warned that a full-scale Turkish incursion attacking Kurdish rebel bases in northernIraq would have disastrous results",
        "The Iraqi Foreign Minister warned of disastrous consequences if Turkey launched a majorinvasion of Iraq to strike at Kurdish rebels"
    ]

    # Sample sextracted from https://github.com/clinc/uniqueness/blob/master/data/random_1.json with intents == phone 
    utterances = [
        'what number do i dial to reach the bank',
        'what is the telephone number of my bank',
        'Is there a number I can reach my bank at?',
        "What's my bank's phone number?",
        'Show me my banks number.',
        'Please let me know the number of my bank.',
        'What is the phone number for my bank?',
        'What is the contact number for my bank',
        'show me how to contact the bank ?',
        'i need a contact phone number for the bank',
        "Gimme me bank's number",
        "Find my bank's phone number",
        'please tell me the number for my bank',
        'What is the phone number for bank?',
    ]

    inliers_embeddings = embed(utterances)
    outliers_embeddings = embed(noisy_sentences)
    utterances.extend(noisy_sentences)

    config_all,config_best = find_best_parameter(inliers_embeddings,outliers_embeddings,utterances)
    print(f"config_all: {config_all}")
    print(f"config_best: {config_best}")
    
    #X_scores,y_pred,ground_truth = LOF_outlier(inliers_embeddings,outliers_embeddings,n_neighbors=20,contamination=0.1)
    sys.exit()

    mean_tensor1, embeddings1, json_dict_scores1 = get_scores(embed,utterances)

    print()
    print(" *Inject noisy data, samples with different intent* ".center(60,'-'))
    print()

    mean_tensor2, embeddings2, json_dict_scores2 = get_scores(embed,utterances)

    sorted_json = sort_dict_by_value(json_dict_scores2,'euclidean_distance')

    #sort dictionary
    #sorted_dict = dict(sorted(my_dict.items(), key=lambda item: item[1]))

    with open('./sortedclean_data.json', 'w') as fp:
        json.dump(sorted_json, fp)

    with open('./noisy_data.json', 'w') as fp:
        json.dump(json_dict_scores2, fp)

def amadeus_get_scores(embed,utterances,paraphrases,metric=0):
    """
    Computes scores for utterances based on embeddings.

    :args
        embed: an isntance of a Universal Sentence Encoder object,  type: tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject
        utterances: A list of utterances for testing.
        paraphrases: A list of paraphrases
        metric (int): metric to use when comparing similarity, 0 cosine similarity, 1 euclidean distance 

    :prints
        - Average vectors computed from the embeddings.
        - Distances (Euclidean and cosine similarity) between the average vector and the embeddings of the provided utterances.
        - Distances (Euclidean and cosine similarity) between the average vector and the embeddings of a test sentence.
        - extended: True if utterances set eas expanded with new candidate, False otherwise
    """

    utterance_embeddings = embed(utterances)

    mean_tensor = get_mean_average_vector(utterance_embeddings)

    paraphrase_embeddings = embed(paraphrases)
    
    outlier_paraphrases = []
    json_dict_scores = dict()

    extended = False

    for i in range(len(paraphrases)):
        euclidean_score = compute_euclidean_distance(mean_tensor,paraphrase_embeddings[i])
        cosine_score = compute_cosine_similarity(mean_tensor,paraphrase_embeddings[i])
        #print_scores(utterances[i],euclidean_score,cosine_score)
        json_dict_scores[paraphrases[i]] = {
            'euclidean_distance': euclidean_score,
            'cosine_similarity': cosine_score
        }

        if metric == 0:
            if cosine_score>0.5 and cosine_score<0.97:
                utterances.append(paraphrases[i])
                extended = True
            else:
                outlier_paraphrases.append(paraphrases[i])
        else:
            if euclidean_score>0.2 and euclidean_score<1.1:
                utterances.append(paraphrases[i])
                extended = True
            else:
                outlier_paraphrases.append(paraphrases[i])
    
    utterances = set(utterances)#to remove duplicates,  maybe current paraphrase is already in the list of utterances
    utterances = list(utterances)#convert to list to avoid error in other part of the code

    return utterances, outlier_paraphrases,json_dict_scores,extended

def amadeus_get_scores_BERT_score(scorer,utterances,paraphrases,metric=0):
    """
    Calculate similarity scores using BERT_score between paraphrases and seed utterances.

    This function is similar to amadeus_get_scores, but uses BERT_score as the similarity measure
    instead of Language Model based embeddings (BERT, USE or GPT).
    Instead of calculating the mean vector of the cluster, then calculating the cosine similarity
    between the embeddings of each candidate and the mean vector, in this function the entire cluster
    is used as input to BERT_score as a multiple reference. So in this function, we avoid calculating
    an average vector of the cluster's embeddings.
    This script is inspired by the BERTScore tutorial https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb

    :args
        scorer: A BERT_score model, a <class 'bert_score.scorer.BERTScorer'> object.
        utterances: A list of seed utterances (List[str]).
        paraphrases: A list of paraphrases (List[str]).
        metric: Which metric to use when assessing the outlier.
                0: BERT-score Precision score.
                1: BERT-score Recall score.
                2: F1 score.
                Default is 0.

    :return
        A tuple containing the following elements:
            - A list of seed utterances with possibly extended paraphrases (List[str]).
            - A list of paraphrases that are considered outliers (List[str]).
            - A dictionary containing similarity scores for each paraphrase (Dict[str, Dict[str, float]]).
            - A boolean value indicating whether any paraphrase was extended (bool).
    
    :examples
        >>> utterances = [['I am proud of you.', 'I love lemons.', 'Go go go.']]
        >>> paraphrases = ['I like lemons.']
        >>> Precision, Recall, F1 = scorer.score(single_cands, multi_refs, lang="en", rescale_with_baseline=True)
    """
    
    outlier_paraphrases = []
    json_dict_scores = dict()

    extended = False
    
    for i in range(len(paraphrases)):

        precision_mul, recall_mul, f1_mul = scorer.score( [paraphrases[i]], [utterances])#return P, R and F1 as tensor object don't forget to get value using item()

        json_dict_scores[paraphrases[i]] = {
            'Precision_Score': precision_mul.item(),
            'Recall_Score': recall_mul.item(),
            'F1_Score': f1_mul.item()
        }

        if metric == 0:
            if precision_mul>0.6 and precision_mul<0.97:
                utterances.append(paraphrases[i])
                extended = True
            else:
                outlier_paraphrases.append(paraphrases[i])
        elif metric == 1:
            if recall_mul>0.6 and recall_mul<<0.97:
                utterances.append(paraphrases[i])
                extended = True
            else:
                outlier_paraphrases.append(paraphrases[i])
        else:
            if f1_mul>0.6 and f1_mul<0.97:
                utterances.append(paraphrases[i])
                extended = True
            else:
                outlier_paraphrases.append(paraphrases[i])
    
    utterances = set(utterances)#to remove duplicates,  maybe current paraphrase is already in the list of utterances
    utterances = list(utterances)#convert to list to avoid error in other part of the code

    return utterances, outlier_paraphrases,json_dict_scores,extended

def run_outlier_with_USE(metric = 0):
    """
    Run outlier detection on the amadeus-dataset-v6.json using Universal Sentence Encoder (USE) embeddings and Euclidean distance.

    This function performs outlier detection on the amadeus-dataset-v6.json using the Universal Sentence Encoder (USE)
    embeddings and the Euclidean distance metric. It first parses command-line arguments to obtain the path to the JSON file
    containing the dataset.

    :args
        -f, --file_path (str): Path to the JSON file containing the dataset. (required)
        metric (int): metric to use when comparing similarity, 0 cosine similarity, 1 euclidean distance
        
    :returns
        None

    :raises
        FileNotFoundError: If the specified JSON file is not found.

    :examples
        >>> run_outlier_with_USE()
    """
    parser = argparse.ArgumentParser(description='Load JSON file')
    parser.add_argument('-f','--file_path', type=str, help='Path to the JSON file',required=True)
    args = parser.parse_args()

    # Load pre-trained universal sentence encoder model
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # prettify_data(args.file_path)

    # a = generate_summary(args.file_path)
    # plot_his_corpus_summary(a)

    with open(args.file_path, 'r') as file:
        json_data = json.load(file)
    
    metric = 0

    result = dict()
    result['metadata'] = "This file contain the result of the outlier detection experiment applied on the amadeus-dataset-v6.json, we applied the euclidean distance to detect outlier."
    utterances = []# add base_utt and test_utt item
    paraphrases = []# add generated_utt
    for k,v in tqdm( json_data.items()):
        base_utt = v['base_utt']
        utterances.append(base_utt)

        test_utt = v['test_utt']
        utterances.extend(test_utt)
        
        generated_utt = v['generated_utt']
        current_prompt = dict()
        for generated_utt_index, field in enumerate(generated_utt):
            paraphrases = field['utterances']
            new_utterances, outlier_paraphrases, json_dict_scores,extended = amadeus_get_scores(embed,utterances.copy(),paraphrases,metric = metric)

            current_prompt[generated_utt_index] = {
                'seed_utterances': utterances.copy(),
                'len_seed_utterances': len(utterances),
                'seed_utterances_extension': extended,
                'new_seed_utterances': new_utterances,
                'outlier_paraphrases': outlier_paraphrases,
                'number_outlier_paraphrases': len(outlier_paraphrases),
                'outlier scores': json_dict_scores
            }
        result[k] = current_prompt
        utterances.clear()

    # Create the filename with the current time
    filename = f"outlier-detection-amadeus-dataset-v6-euclidean"
    save_data_as_json(result,filename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load JSON file')
    parser.add_argument('-f','--file_path', type=str, help='Path to the JSON file',required=True)
    args = parser.parse_args()

    # run_outlier_with_USE()
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    metric = 2# 0 for precision, 1 for Recall and 2 for F1

    with open(args.file_path, 'r') as file:
        json_data = json.load(file)
    
    result = dict()
    result['metadata'] = "This file contain the result of the outlier detection experiment applied on the amadeus-dataset-v6.json, we applied the euclidean distance to detect outlier."
    utterances = []# add base_utt and test_utt item
    paraphrases = []# add generated_utt
    for k,v in tqdm( json_data.items()):
        base_utt = v['base_utt']
        utterances.append(base_utt)

        test_utt = v['test_utt']
        utterances.extend(test_utt)

        utterances = list(map(str.strip, utterances))#removes any leading, and trailing whitespaces
        
        generated_utt = v['generated_utt']
        current_prompt = dict()
        for generated_utt_index, field in enumerate(generated_utt):
            paraphrases = field['utterances']
            paraphrases = list(map(str.strip, paraphrases))#removes any leading, and trailing whitespaces
            new_utterances, outlier_paraphrases, json_dict_scores,extended = amadeus_get_scores_BERT_score(scorer,utterances.copy(),paraphrases,metric)

            current_prompt[generated_utt_index] = {
                'seed_utterances': utterances.copy(),
                'len_seed_utterances': len(utterances),
                'seed_utterances_extension': extended,
                'new_seed_utterances': new_utterances,
                'outlier_paraphrases': outlier_paraphrases,
                'number_outlier_paraphrases': len(outlier_paraphrases),
                'outlier scores': json_dict_scores
            }
        result[k] = current_prompt
        utterances.clear()

    # Create the filename with the current time
    filename = f"outlier-detection-amadeus-dataset-v6-bert-score"

    save_data_as_json(result,filename)