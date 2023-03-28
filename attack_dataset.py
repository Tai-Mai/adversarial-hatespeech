import argparse
import torch
from transformers import AutoTokenizer #, AutoModelForSequenceClassification
from pretrained.models import *
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.eval import predict
from utils.attack import attack
from utils.data import (format_dataset_file, load_data, fast_forward, 
                        save_adversarial_examples, prediction_to_dataset_file)
from tqdm import tqdm
import string
import os
from explain import lime_explain


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
    parser = argparse.ArgumentParser(
            description="Create adversarial attacks on HateXplain"
    )
    parser.add_argument(
            "--split", 
            help="Which dataset split to use. {test, train, val}"
    )
    parser.add_argument(
            "--permissible_subs", 
            help="Permissible substitutions. {no-letters, all-chars}. Default: no-letters",
            default="no-letters"
    )
    parser.add_argument(
            "--substitution_file", 
            help="Substitution dictionary that's generated after running main.py and analyze.py",
            default=None
    )
    args = parser.parse_args()

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

    # Load dataset
    dataset_file = "data/dataset.json"
    
    split = args.split

    # Characters for substitution
    if args.permissible_subs == "no-letters":
        permissible_substitutions = string.punctuation + string.digits
        # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789
    elif args.permissible_subs == "all-chars":
        permissible_substitutions = string.printable
        # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    else:
        raise ValueError("`permissible_subs` either `no-letters` or `all-chars`.")


    # Only does something when using the original dataset file by the authors
    prepare_dataset_file(dataset_file, model, tokenizer)

    print("Loading dataset...")
    dataset = load_data(dataset_file, split=split)
    num_datapoints = sum(1 for _ in dataset)
    # reset dataset generator
    dataset = load_data(dataset_file, split=split)

    # JSON file that will contain the found adversarial examples
    target_file = f"data/attacks_{split}_{args.permissible_subs}.json"

    # Fast-forward the dataset to the last attacked datapoint in case target_file 
    # exists. 
    if os.path.exists(target_file) and args.substitution_file is None:
        print("Fast-forwarding...")
        num_skipped = fast_forward(dataset, target_file)
    else:
        num_skipped = 0

    # cumulative_success_rate = 0
    # cumulative_success_rate_in_top_k = 0

    print("Attacking dataset...")
    if args.substitution_file is None:
        for post in tqdm(dataset, total=num_datapoints-num_skipped):    
            # 1142 is number of abusive datapoints in the test set

            original_text = TreebankWordDetokenizer().detokenize(post["post_tokens"])

            results = attack(original_text, model, tokenizer, permissible_substitutions)

            save_adversarial_examples(post["post_id"], results, target_file)
    else:
        print("Substitutions file was provided:")
        print(args.substitution_file)
        print("Using look-up substitutions for attacks...")
        source_split = args.substitution_file.split("_")[1]
        permissible_substitutions = args.substitution_file.split("_")[2].split(".")[0]

        with open(args.substitution_file) as f:
            subs_dict = json.load(f)

        successful = 0
        for post in tqdm(dataset, total=num_datapoints):    
        # for post in dataset:
            text = post["post_tokens"]
            # Find the token in the current post that was attacked the most.  
            # That token will be our naive attack vector.
            highscore = 0
            victim_token = None
            for i, token in enumerate(text):
                try:
                    if subs_dict[token]["_num_attacks"] > highscore:
                        highscore = subs_dict[token]["_num_attacks"] 
                        victim_token = token
                        victim_index = i
                except KeyError:    # token not in substitution dictionary
                    continue

            if victim_token is None:
                continue
            else:
                # Pick the most successful substitute token
                substitute_token = list(subs_dict[victim_token])[1]
                # Substitute the victim_token with the substitute_token
                text[victim_index] = substitute_token
                text = TreebankWordDetokenizer().detokenize(text)
                probabilities = predict(text, model, tokenizer)
                if probabilities[1] < 0.5:  # if abusive score is below 0.5
                    successful += 1
                else:
                    text_explained = lime_explain(text, model, tokenizer)
                    with open(f"data/{split}_{source_split}_{permissible_substitutions}_unsuccessful.txt", "a") as f:
                        f.write("Text:\n")
                        f.write(text)
                        f.write("\n\nExplanation for abusive classification:\n")
                        f.write(str(text_explained))
                        f.write("\n\n\n")

        success_rate = successful / num_datapoints
        
        with open(f"data/{split}_{source_split}_{permissible_substitutions}_unsuccessful.txt", "a") as f:
            f.write("Attack success rate: {}/{} = {:.4f}".format(successful, 
                                                           num_datapoints, 
                                                           success_rate))


    print("Done.")

if __name__ == "__main__":
    main()
