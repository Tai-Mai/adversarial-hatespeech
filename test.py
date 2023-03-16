import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pretrained.models import *
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.eval import evaluate
from utils.attack import attack

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
    """
    Generates samples from the dataset. Only yields datapoints that are abusive
    (offensive or hatespeech).

    Parameters
    ----------
    ids : List(str)
        List of post_ids of which to return the datapoints

    Yields
    ------
    data[i] : Dict
        Datapoint
    """
    with open("data/dataset.json") as data_file:
        data = json.load(data_file)
    for i in ids:
        num_annotators = len(data[i]["annotators"])
        num_normal = 0
        for annotator in data[i]["annotators"]:
            if annotator["label"].lower() == "normal":
                num_normal += 1
        if num_normal < num_annotators/2:
            yield data[i]


counter = 0
batchsize = 8
print("Attacking dataset...")
for post in dataset(test_ids):
    # if counter == 5:
    #     break
    # counter += 1

    text = TreebankWordDetokenizer().detokenize(post["post_tokens"])

    # attacks = attack("You are a nigger", model, tokenizer)
    # print(attacks)

    # probabilities = evaluate(["this is a test", "this is a tast"], model, tokenizer)
    # probabilities = evaluate(attacks, model, tokenizer)
    # print(probabilities)
    # print(f"Normal: {probabilities[0][0]}\nHatespeech: {probabilities[0][1]}\n\n")
    # print(f"Normal: {probabilities[1][0]}\nHatespeech: {probabilities[1][1]}\n\n")

    attacked = ["You are a n{gger who eats pie.",
                "You are a ni@ger who eats pie.",
                "You are a n{gger who eats pie.",
                "You are a n~gger who eats pie.",
                "You are a nigeer who eats pie."]
    probabilities = evaluate(attacked, model, tokenizer)
    print("1st column: non-abusive. 2nd column: abusive")
    print(probabilities)

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

print("Done.")
