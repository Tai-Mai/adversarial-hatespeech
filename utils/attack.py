from typing import Union
import transformers
import string
import torch
from utils.eval import evaluate

def attack(original_text, model, tokenizer, subs=1, top_k=5):
    """
    Return adversarial examples

    Parameters
    ----------
    original_text : str
        Text to be attacked/modified.
    model : transformers.AutoModelForSequenceClassification
        Victim model, trained HateXplain model
    tokenizer : transformers.AutoTokenizer
        Tokenizer from trained HateXplain model
    subs : int
        Number of character substitutions. 
        Default: 1
    top_k : int
        Return this many of the best candidates. Best is determined by how much
        they influence the probability scores
        Default: 5

    Returns
    -------
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
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Compute probabilities prior to the attacks
    # inputs = tokenizer(
    #     text, 
    #     return_tensors="pt", 
    #     padding=True
    # ).to(device)
    # prediction_logits, _ = model(
    #     input_ids=inputs['input_ids'],
    #     attention_mask=inputs['attention_mask']
    # )
    # softmax = torch.nn.Softmax(dim=1)
    # prior_probabilities = softmax(prediction_logits)
    # prior_hatespeech_probability = prior_probabilities[0][1]

    prior_abusive_probability = evaluate(original_text, model,
                                         tokenizer)[0][1]
    # convert candidate score (torch.Tensor) to float/double
    prior_abusive_probability = prior_abusive_probability.item()

    # Generate attacks
    candidate_scores = {}
    for i, char in enumerate(original_text):
        for candidate in generate_candidates(original_text, i):
            candidate_probability = evaluate(candidate, model,
                                             tokenizer)[0][1].item()
            
            candidate_score = prior_abusive_probability - candidate_probability
            # higher score is better
            candidate_scores[candidate] = candidate_score
            print(f"candidate_score: {candidate_score}")

            # print("----------------------------------")
            # print(f"i: {i}")
            # print(f"char: {char}")
            # print(f"candidate: {candidate}")
            # print(f"score: {candidate_score}")
            # print("----------------------------------")


    sorted_candidate_scores = dict(sorted(candidate_scores.items(), 
                                   key=lambda item: item[1], 
                                   reverse=True))
    attacks = list(sorted_candidate_scores)[:top_k]
    attacks_scores = list(sorted_candidate_scores.values())[:top_k]

    results = {
        "original" : {
            "text" : original_text,
            "abusive_probability" : prior_abusive_probability
        },
        "attacks" : []
    }

    for attack, score in zip(attacks, attacks_scores):
        # print(f"{score}: {attack}")
        print(f"candidate_score: {candidate_score}")
        result = {
            "text" : attack,
            "abusive_probability" : prior_abusive_probability - score
        }
        results["attacks"].append(result)

    # print(f"ATTACKS: {attacks}")
    # print(f"ATTACKS SCORES: {attacks_scores}")

    # print(results)

    return results


def generate_candidates(text, i):
    """
    Substitute a character in the text with every possible substitution 

    Parameters
    ----------
    text : str
        Text to be attacked/modified.
    i : int
        Index of character to be substituted

    Yields
    ------
    candidate : 
        List of the `top_k` attacks on the input text
    """

    permissible_substitutions = string.printable
    # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

    for substitution_char in permissible_substitutions:
        if substitution_char == text[i]:
            continue
        candidate = list(text)
        candidate[i] = substitution_char 
        candidate = "".join(candidate)
        yield candidate
