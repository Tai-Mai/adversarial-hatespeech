from typing import Union
import torch


def evaluate(text, model, tokenizer, return_tensor=False):
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
        `probabilities[0]` for `normal` and 
        `probabilities[1]` for `abusive` (offensive or hatespeech).
    """
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

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
        # convert tensor to list
        if return_tensor:
            probabilities = probabilities.cpu().detach().numpy()
        else:
            probabilities = probabilities.squeeze().tolist()
        # print(probabilities)
        # print(f"Normal: {probabilities[0][0]}\nHatespeech: {probabilities[0][1]}\n\n")

        return probabilities
