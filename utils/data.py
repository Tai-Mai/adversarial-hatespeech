import json
import os
import torch
from shutil import copyfile
from tqdm import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer
from .eval import evaluate
# from ..pretrained.models import Model_Rational_Label
# from transformers import AutoTokenizer
# from .eval import evaluate


def load_data(dataset_file, only_abusive=True, split=None):
    """
    Generates samples from the dataset. Only yields datapoints that are abusive
    (offensive or hatespeech)!

    Parameters
    ----------
    dataset_file : str
        Path/name of dataset file.
    split : str
        train, test, dev, or None (default).

    Yields
    ------
    data[i] : Dict
        Datapoint
    """

    with open(dataset_file) as data_file:
        data = json.load(data_file)

    if split is not None:
        # Load only ids from a specific split (train, dev, or test)
        with open("data/post_id_divisions.json") as f:
            splits = json.load(f)
            ids = splits[split]
    else:
        # Load all ids
        ids = data.keys()

    if only_abusive:
        for i in ids:
            num_annotators = 3
            num_normal = 0
            for annotator in data[i]["annotators"]:
                if annotator["label"].lower() == "normal":
                    num_normal += 1
            # Return datapoint if annotators and model agree that it's abusive
            if num_normal < num_annotators/2 and data[i]["prediction"] == "abusive":
                yield data[i]
    else:
        for i in ids:
            yield data[i]


def fast_forward(dataset, filename):
    """
    If `target_file` exists (and is incomplete), jump to where it was left off.
    If no `target_file` exists, do nothing.

    Parameters
    ----------
    dataset : generator
        Dataset that should be fast-forwarded.
    filename : str
        Name/location of the target file that contains the adversarial examples.

    Return
    ------
    num_skipped : int
        Number of skipped datapoints.
    """
    num_skipped = 0
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_datapoints = json.load(f)

        for i in range(len(existing_datapoints)):
            num_skipped += 1
            next(dataset)

    return num_skipped


def save_adversarial_examples(post_id, results, filename):
    """
    Save adversarial examples to the passed JSON file.

    Parameters
    ----------
    post_id : str
        post_id of the post in the dataset.
    results : dict
        Dictionary of the `top_k` attacks on the input text
        ```
        results = {
            "original" : {
                "text" : original text,
                "abusive_probability" : probability before attack
            },
            "top_k_attacks" : [
                {
                    "text" : attack1,
                    "abusive_probability" : probability after attack
                },
                {
                    "text" : attack2,
                    "abusive_probability" : probability after attack
                },
                ...
            ],
            stats = {
                "text_length" : Character length of text
                "num_attacks" : Total number of attacks
                "num_successful" : Total number of successful attacks 
                "num_successful_top_k" : Number of successful attacks in top k
                "success_rate" : Success rate among all attacks
                "success_rate_top_k" : Success rate among top k attacks
            }
        }
        ```
    filename : str
        Under which filename to save. Can pre-exist, in which case lines are 
        appended to the existing file.
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_results = json.load(f)
    else:
        # create new json
        existing_results = {}

    results = {post_id : results}

    # merge existing data with new data
    full_results = {**existing_results, **results}
    with open(filename, "w") as f:
        json.dump(full_results, f, indent=4)


def format_dataset_file(dataset_file):
    """
    Creates indents in the original dataset file for easier searchability.

    Parameters
    ----------
    original_file : str
        Name/path of original dataset JSON file.
    target_file : str
        Name/path of target dataset JSON file.
    """
    path_split = os.path.split(dataset_file)
    original_file = os.path.join(path_split[0], "original_"+path_split[1])
    copyfile(dataset_file, original_file)

    with open(dataset_file) as data_file:
        data = json.load(data_file)

    with open(dataset_file, "w") as f:
        json.dump(data, f, indent=4)


def prediction_to_dataset_file(dataset_file, model, tokenizer):
    """
    Adds the model's vanilla prediction (no adversarial attacks) to the dataset 
    file.

    Parameters
    ----------
    dataset_file : str
        Name/path of dataset JSON file.
    model : pretrained.models.Model_Rational_Label
        Hatespeech detection model.
    tokenizer : nltk.tokenize.treebank.TreebankWordDetokenizer
        Detokenizes the datapoint text since datapoints are saved as lists of
        tokens in the dataset.

    Returns
    -------
    num_correct : int
        Number of correct predictions. Correct means model predicted "abusive"
        if more than half of the annotators annotated either "offensive" or
        "hatespeech".
    """
    with open(dataset_file) as f:
        data = json.load(f)

    dataset = load_data(dataset_file, only_abusive=False)
    num_correct = 0
    for post in tqdm(dataset, total=20148):    # 20148 samples in entire dataset
        post_id = post["post_id"]
        text = TreebankWordDetokenizer().detokenize(post["post_tokens"])
        probability_normal, probability_abusive = evaluate(text, 
                                                           model, 
                                                           tokenizer)
        if probability_abusive > probability_normal:
            prediction = "abusive"
        else:
            prediction = "normal"
        
        data[post_id]["prediction"] = prediction

        num_annotators = 3
        num_normal = 0
        for annotator in data[post_id]["annotators"]:
            if annotator["label"].lower() == "normal":
                num_normal += 1
        if num_normal < num_annotators/2:
            annotation = "abusive"
        else: 
            annotation = "normal"
        if prediction == annotation:
            num_correct += 1

    with open(dataset_file, "w") as f:
        print("Updating file...")
        json.dump(data, f, indent=4)

    print(f"Correctly classified: {num_correct}/20148 =", 
           "{:.2f}".format(num_correct/20148))


def main():
    pass


if __name__ == "__main__":
    main()
