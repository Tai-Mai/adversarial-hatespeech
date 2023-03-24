import torch
from transformers import AutoTokenizer #, AutoModelForSequenceClassification
from pretrained.models import *
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.eval import evaluate
from lime.lime_text import LimeTextExplainer
import json
from tqdm import tqdm
import gc


def lime_explain(model, tokenizer, attacks_file, top_k=5, num_features=5):
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
    
    with open(attacks_file, "r") as f:
        attacks = json.load(f)

    with torch.no_grad():
        for post_id, results in tqdm(attacks.items(), total=len(attacks)):
            original = results["original"]["text"]

            # text_list = [original]
            # for k in range(top_k):
            #     text_list.append(results["attacks"][k]["text"])

            # exp = explainer.explain_instance(
            #         text_list, 
            #         lambda text: evaluate(text, model, tokenizer,
            #                               return_tensor=True),
            #         num_features=num_features
            # )


            exp_original = explainer.explain_instance(
                    original, 
                    lambda text: evaluate(text, model, tokenizer,
                                          return_tensor=True),
                    num_features=num_features,
                    num_samples=100
            )
            # print(torch.cuda.memory_summary(device=0))
            # print("\nExplanation of original text:\n", exp_original.as_list())
            attacks[post_id]["original"]["explanation"] = exp_original.as_list()

            for k, attack in enumerate(results["top_k_attacks"][:top_k]):
                exp_attack = explainer.explain_instance(
                        attack["text"],
                        lambda text: evaluate(text, model, tokenizer,
                                              return_tensor=True),
                        num_features=num_features,
                        num_samples=100
                )
                # print(torch.cuda.memory_summary(device=0))
                # print("Explanation of attacked text:\n", exp_attack.as_list())
                attacks[post_id]["top_k_attacks"][k]["explanation"] = exp_attack.as_list()

            with open(attacks_file, "w") as f:
                json.dump(attacks, f, indent=4)


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

    target_file = "data/adversarial_examples_no-letters.json"

    lime_explain(model, tokenizer, target_file)


if __name__ == "__main__":
    main()
