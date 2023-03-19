import torch
from transformers import AutoTokenizer #, AutoModelForSequenceClassification
from pretrained.models import *
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.eval import evaluate
from utils.attack import attack
from utils.data import (format_dataset_file, load_data, fast_forward, 
                        save_adversarial_examples, prediction_to_dataset_file)
from tqdm import tqdm


def prepare_dataset_file(dataset_file, model, tokenizer):
    """
    Checks if the dataset file is the original one by the authors. In that case,
    format it so it's not all just one single long line and write the model's
    prediction to the dataset file.
    """
    with open(dataset_file) as f:
        for line_index, _ in enumerate(f):
            pass
        if line_index == 0:
            # Original file from the authors with all data in one single line. 
            # In that case, format the dataset file.
            print("Formatting dataset file...")
            format_dataset_file(dataset_file)
            print("Writing vanilla predictions to dataset file...")
            prediction_to_dataset_file(dataset_file, model, tokenizer)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
    )
    print(f"Loading model...")
    model = Model_Rational_Label.from_pretrained(
        "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
    )
    model = model.to(device)
    model.eval()

    # Load test dataset
    dataset_file = "data/dataset.json"

    # Only does something when using the original dataset file by the authors
    prepare_dataset_file(dataset_file, model, tokenizer)

    print("Loading dataset...")
    dataset = load_data(dataset_file, split="test")
    num_datapoints = sum(1 for _ in dataset)
    # reset dataset generator
    dataset = load_data(dataset_file, split="test")

    # JSON file that will contain the found adversarial examples
    target_file = "data/adversarial_examples.json"
    # Fast-forward the dataset to the last attacked datapoint in case target_file 
    # exists. If it doesn't, `fast_forward` will do nothing and `num_skipped` = 0
    print("Fast-forwarding...")
    num_skipped = fast_forward(dataset, target_file)

    print("Attacking dataset...")
    for post in tqdm(dataset, total=num_datapoints-num_skipped):    
        # 1142 is number of abusive datapoints in the test set

        original_text = TreebankWordDetokenizer().detokenize(post["post_tokens"])

        results = attack(original_text, model, tokenizer)
        print(results)

        save_adversarial_examples(post["post_id"], results, target_file)

        # attacked = ["You are a n{gger who eats pie.",
        #             "You are a ni@ger who eats pie.",
        #             "You are a n{gger who eats pie.",
        #             "You are a n~gger who eats pie.",
        #             "You are a nigeer who eats pie."]
        # probabilities = evaluate(attacked, model, tokenizer)
        # print("1st column: non-abusive. 2nd column: abusive")
        # print(probabilities)

    print("Done.")

if __name__ == "__main__":
    main()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Device: {device}\n")

    # print(f"Loading tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
    # )
    # print(f"Loading model...")
    # model = Model_Rational_Label.from_pretrained(
    #     "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
    # )
    # model = model.to(device)
    # model.eval()

    # # Load test dataset
    # dataset_file = "data/dataset.json"

    # prediction_to_dataset_file(dataset_file, model, tokenizer)
    # print("Done.")
