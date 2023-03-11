from typing import Union
import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from pretrained.models import *
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# tokenizer = AutoTokenizer.from_pretrained(
#     "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
# )
# model = Model_Rational_Label.from_pretrained(
#     "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
# )
# model = model.to(device)

def eval(text, model, tokenizer):
    """
    Get model's prediction on a text.

    Parameters
    ----------
    text : Union[str, list]
        Text to be classified. Either a single string or a list of strings
    model : transformers.AutoModelForSequenceClassification
        Trained HateXplain model
    tokenizer : transformers.AutoTokenizer
        Tokenizer from trained HateXplain model

    Returns
    -------
    probabilities : torch.Tensor
        If text is only one string, then get probabilities with
        `probabilities[0][0]` for `normal` and 
        `probabilities[0][1]` for `hatespeech`.
        If text is multiple strings in a list, then get probabilities with
        `probabilities[i][0]` and `probabilities[i][1]`, respectively, where
        `i` is the sample in the batch.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True
    ).to(device)
    prediction_logits, _ = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(prediction_logits)
    # print(f"Normal: {probabilities[0][0]}\nHatespeech: {probabilities[0][1]}\n\n")

    return probabilities



