import json
import os


def load_data(split):
    """
    Generates samples from the dataset. Only yields datapoints that are abusive
    (offensive or hatespeech).

    Parameters
    ----------
    split : str
        train, test, or dev

    Yields
    ------
    data[i] : Dict
        Datapoint
    """
    with open("data/post_id_divisions.json") as splits:
        data = json.load(splits)
        ids = data[split]

    with open("data/dataset.json") as data_file:
        data = json.load(data_file)

    for i in ids:
        num_annotators = len(data[i]["annotators"])
        num_normal = 0
        for annotator in data[i]["annotators"]:
            if annotator["label"].lower() == "normal":
                num_normal += 1
        if num_normal < num_annotators/2:
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
    Save adversarial examples to a .json file.

    Parameters
    ----------
    post_id : str
        post_id of the post in the dataset.
    results : dict
        List of the `top_k` attacks on the input text
        ```
        results = {
            "original" : {
                "text" : original text,
                "abusive_probability" : probability before attack
            },
            "attacks" : [
                {
                    "text" : attack1,
                    "abusive_probability" : probability after attack
                },
                {
                    "text" : attack2,
                    "abusive_probability" : probability after attack
                },
                ...
            ]
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


def main():
    pass

if __name__ == "__main__":
    main()
