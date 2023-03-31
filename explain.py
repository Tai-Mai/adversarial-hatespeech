import torch
from transformers import AutoTokenizer #, AutoModelForSequenceClassification
from pretrained.models import *
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.eval import predict
from lime.lime_text import LimeTextExplainer
import json
from tqdm import tqdm
import gc
import argparse


def lime_explain(text, model, tokenizer, top_k=5, num_features=5):
    """
    Reads the file containing the attacks, gets rationale explanations using 
    LIME, and adds them to the file in-place.

    Parameters
    ----------
    model : transformers.AutoModelForSequenceClassification
        Victim model, trained HateXplain model
    tokenizer : transformers.AutoTokenizer
        Tokenizer from trained HateXplain model
    attacks_file : str
        Name/path of JSON file containing the adversarial attacks.
    top_k : int
        How many attacks to explain
        Default: 5
    num_features : int
        Number of tokens to include in explanations.
        Default: 5

    Returns
    -------
    explanations : List(Tuples)
        List of tuples of tokens and their respective score influence.
    """

    explainer = LimeTextExplainer(
            class_names=["normal", "abusive"],
            bow=False
    )

    text_explained = explainer.explain_instance(
            text, 
            lambda txt: predict(txt, model, tokenizer, return_tensor=True),
            num_features=num_features,
            num_samples=100
    )

    return text_explained.as_list()
    


def main():
    parser = argparse.ArgumentParser(
            description="Explain adversarial attacks with LIME"
    )
    parser.add_argument(
            "--attacks_file", 
            help="JSON file containing the generated adversarial attacks",
            default=None
    )
    parser.add_argument(
            "--top_n", 
            help="How many of the top attacks to explain. Must be not above top_k. Default: k=5, n=3",
            default=5
    )
    args = parser.parse_args()
    # target_file = "data/adversarial_examples_all-chars.json"
    attacks_file = args.attacks_file
    top_n = int(args.top_n)

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


    with open(attacks_file, "r") as f:
        attacks = json.load(f)

    with torch.no_grad():
        for post_id, results in tqdm(attacks.items(), total=len(attacks)):
            original = results["original"]["text"]

            # exp_original = explainer.explain_instance(
            #         original, 
            #         lambda text: predict(text, model, tokenizer,
            #                              return_tensor=True),
            #         num_features=num_features,
            #         num_samples=100
            # )

            exp_original = lime_explain(original, model, tokenizer)
            # attacks[post_id]["original"]["explanation"] = exp_original.as_list()
            attacks[post_id]["original"]["explanation"] = exp_original

            for n, attack in enumerate(results["top_k_attacks"][:top_n]):
                # exp_attack = explainer.explain_instance(
                #         attack["text"],
                #         lambda text: predict(text, model, tokenizer,
                #                              return_tensor=True),
                #         num_features=num_features,
                #         num_samples=100
                # )
                exp_attack = lime_explain(attack["text"], model, tokenizer)
                # attacks[post_id]["top_k_attacks"][k]["explanation"] = exp_attack.as_list()
                attacks[post_id]["top_k_attacks"][n]["explanation"] = exp_attack

            with open(attacks_file, "w") as f:
                json.dump(attacks, f, indent=4)


if __name__ == "__main__":
    main()
