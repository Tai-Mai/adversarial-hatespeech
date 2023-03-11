import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pretrained.models import *
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.eval import eval
from utils.attack import attack

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
)
model = Model_Rational_Label.from_pretrained(
    "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
)
model = model.to(device)


# inputs = tokenizer(['He is a great guy', 'He is a greeeeeet guy'],
#         return_tensors="pt", padding=True).to(device)
# print(inputs['input_ids'])
# prediction_logits, _ = model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
# softmax = torch.nn.Softmax(dim=1)
# probs = softmax(prediction_logits)
# print(f"Normal: {probs[0][0]}\nHatespeech: {probs[0][1]}")
# print(f"Normal: {probs[1][0]}\nHatespeech: {probs[1][1]}")

# Load test dataset
with open("data/post_id_divisions.json") as splits:
    data = json.load(splits)
    test_ids = data["test"]


def dataset(ids):
    with open("data/dataset.json") as data_file:
        data = json.load(data_file)
    for i in ids:
        yield data[i]


counter = 0
batchsize = 8
for post in dataset(test_ids):
    # if counter == 5:
    #     break
    # counter += 1

    text = TreebankWordDetokenizer().detokenize(post["post_tokens"])

    attacks = attack(text, model, tokenizer)
    print(attacks)

    probabilities = eval(attacks, model, tokenizer)
    # probabilities = eval(["this is a test", "this is a tast"], model, tokenizer)
    print(probabilities)
    # print(f"Normal: {probabilities[0][0]}\nHatespeech: {probabilities[0][1]}\n\n")
    # print(f"Normal: {probabilities[1][0]}\nHatespeech: {probabilities[1][1]}\n\n")

    # ATTACK HERE
    # batch = attack(detokenized)

    # inputs = tokenizer(detokenized, return_tensors="pt", padding=True).to(device)
    # prediction_logits, _ = model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
    # softmax = torch.nn.Softmax(dim=1)
    # probs = softmax(prediction_logits)
    # print(f"Normal: {probs[0][0]}\nHatespeech: {probs[0][1]}\n\n")
    # print(f"Normal: {probs[1][0]}\nHatespeech: {probs[1][1]}\n\n")
    #
    break

    # print("------------------")
    # print(post["post_id"])
    # print(post["annotators"][0]["label"])
    # print(TreebankWordDetokenizer().detokenize(post["post_tokens"]))
