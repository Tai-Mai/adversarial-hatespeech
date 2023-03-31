from typing import Union
import transformers
import string
import torch
from utils.eval import predict
from explain import lime_explain

def attack(original_text, model, tokenizer, permissible_substitutions,
           lime=False, top_k=5):
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
    lime : bool
        Whether to use lime for choosing a victim token. If True, limits the
        attack area to that token.
    top_k : int
        Return this many of the best candidates. Best is determined by how much
        they influence the probabilities
        Default: 5

    Returns
    -------
    results : dict
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
                "success_rate" : Success rate among all attacks
            }
        }
        ```
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    prior_abusive_probability = predict(original_text, model,
                                         tokenizer)[1]
    # Generate attacks
    candidate_probabilities = {}

    if lime:
        text_explained = lime_explain(original_text, model, tokenizer)
        print(text_explained)
        victim_token = text_explained[0][0]
        print(victim_token, '\n')
        start_index = original_text.index(victim_token)
        end_index = start_index + len(victim_token)
    else: 
        start_index = 0
        end_index = len(original_text)

    for i, char in enumerate(original_text[start_index:end_index]):
        if char in string.whitespace: 
            continue
        for candidate in generate_candidates(original_text, i+start_index, permissible_substitutions):
            candidate_probability = predict(candidate, model,
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

    results = {
        "original" : {
            "text" : original_text,
            "abusive_probability" : prior_abusive_probability
        },
        "top_k_attacks" : [],
        "stats" : {
            "text_length" : len(original_text),
            "num_attacks" : len(original_text) * len(permissible_substitutions),
            "num_successful" : num_successful,
            "success_rate" : success_rate,
        }
    }

    for attack, probability in zip(attacks, attacks_probabilities):
        # print(f"{score}: {attack}")
        result = {
            "text" : attack,
            "abusive_probability" : probability
        }
        results["top_k_attacks"].append(result)

    # print(results)

    return results


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
        String of text after substitution
    """

    for substitution_char in permissible_substitutions:
        if substitution_char == text[i]:
            continue
        candidate = list(text)
        candidate[i] = substitution_char 
        candidate = "".join(candidate)
        yield candidate
