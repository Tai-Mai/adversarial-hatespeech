from typing import Union
import transformers
import string
import torch
from utils.eval import evaluate

def attack(original_text, model, tokenizer, permissible_substitutions, top_k=5):
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
    permissible_substitutions : str
        String containing all permissible substitution characters
    top_k : int
        Return this many of the best candidates. Best is determined by how much
        they influence the probabilities
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
    success_rate : float
        Success rate among all attempted attacks
    success_rate_in_top_k : float
        Success rate among the top k attacks
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    prior_abusive_probability = evaluate(original_text, model,
                                         tokenizer)[1]
    # Generate attacks
    candidate_probabilities = {}
    for i, char in enumerate(original_text):
        for candidate in generate_candidates(original_text, i, permissible_substitutions):
            candidate_probability = evaluate(candidate, model,
                                             tokenizer)[1]
            
            # candidate_score = prior_abusive_probability - candidate_probability
            # higher score is better
            candidate_probabilities[candidate] = candidate_probability

    # sorted_candidate_scores = dict(sorted(candidate_scores.items(), 
    #                                key=lambda item: item[1], 
    #                                reverse=True))
    sorted_candidate_probabilities = dict(sorted(candidate_probabilities.items(), 
                                                 key=lambda item: item[1]))
    attacks = list(sorted_candidate_probabilities)[:top_k]
    attacks_probabilities = list(sorted_candidate_probabilities.values())
    num_successful = sum(prob < 0.5 for prob in attacks_probabilities)
    success_rate = num_successful / (len(original_text) * len(permissible_substitutions))
    attacks_probabilities = attacks_probabilities[:top_k]
    num_successful_in_top_k = sum(prob < 0.5 for prob in attacks_probabilities)
    success_rate_in_top_k = num_successful_in_top_k / top_k

    results = {
        "original" : {
            "text" : original_text,
            "abusive_probability" : prior_abusive_probability
        },
        "attacks" : []
    }

    for attack, probability in zip(attacks, attacks_probabilities):
        # print(f"{score}: {attack}")
        result = {
            "text" : attack,
            "abusive_probability" : probability
        }
        results["attacks"].append(result)

    # print(results)

    return results, success_rate, success_rate_in_top_k


def generate_candidates(text, i, permissible_substitutions):
    """
    Substitute a character in the text with every possible substitution 

    Parameters
    ----------
    text : str
        Text to be attacked/modified.
    i : int
        Index of character to be substituted
    permissible_substitutions : str
        String containing all permissible substitution characters

    Yields
    ------
    candidate : 
        List of the `top_k` attacks on the input text
    """

    for substitution_char in permissible_substitutions:
        if substitution_char == text[i]:
            continue
        candidate = list(text)
        candidate[i] = substitution_char 
        candidate = "".join(candidate)
        yield candidate
