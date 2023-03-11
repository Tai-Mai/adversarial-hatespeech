from typing import Union
import transformers
import string

def attack(text, model, tokenizer, subs=1, top_k=5):
    """
    Return adversarial examples

    Parameters
    ----------
    text : str
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
    attacks : List[str]
        List of the `top_k` attacks on the input text
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

    prior_hatespeech_probability = eval(text, model, tokenizer)[0][1]

    # Generate attacks
    candidate_scores = {}
    for i, char in enumerate(text):
        for candidate in generate_candidates(text, i, model, tokenizer):
            candidate_probability = eval(candidate, model, tokenizer)[0][1]
            
            candidate_score = prior_hatespeech_probability - candidate_probability
            # higher score is better
            candidate_scores[candidate] = candidate_score

    sorted_candidate_scores = dict(sorted(candidate_scores.items(), 
                                   key=lambda item: item[1], 
                                   reverse=True))
    attacks = list(sorted_candidate_scores)[:top_k]
    return attacks


def generate_candidates(text, i, model, tokenizer)
    """
    Substitute a character in the text with every possible substitution 

    Parameters
    ----------
    text : str
        Text to be attacked/modified.
    i : int
        Index of character to be substituted
    model : transformers.AutoModelForSequenceClassification
        Victim model, trained HateXplain model
    tokenizer : transformers.AutoTokenizer
        Tokenizer from trained HateXplain model

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
