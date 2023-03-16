import torch
from transformers import AutoTokenizer #, AutoModelForSequenceClassification
from pretrained.models import *
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.eval import evaluate
from utils.attack import attack
from utils.data import load_data
from utils.data import fast_forward
from utils.data import save_adversarial_examples
from tqdm import tqdm

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

dataset = load_data()

# JSON file that will contain the found adversarial examples
target_file = "data/adversarial_examples.json"
# Fast-forward the dataset to the last attacked datapoint in case target_file 
# exists. If it doesn't, `fast_forward` will do nothing and `num_skipped` = 0
num_skipped = fast_forward(dataset, target_file)

# batchsize = 8
print("Attacking dataset...")
for post in tqdm(dataset, total=1142-num_skipped):    
    # 1142 is number of abusive datapoints in the test set

    # i to resume in the middle of the dataset
    # if counter == 5:
    #     break
    # counter += 1

    original_text = TreebankWordDetokenizer().detokenize(post["post_tokens"])

    attacks = attack(post["post_id"], original_text, model, tokenizer)
    print(attacks)

    save_adversarial_examples(post["post_id"], original_text, attacks, target_file)

    # probabilities = evaluate(["this is a test", "this is a tast"], model, tokenizer)
    # probabilities = evaluate(attacks, model, tokenizer)
    # print(probabilities)
    # print(f"Normal: {probabilities[0][0]}\nHatespeech: {probabilities[0][1]}\n\n")
    # print(f"Normal: {probabilities[1][0]}\nHatespeech: {probabilities[1][1]}\n\n")

    # attacked = ["You are a n{gger who eats pie.",
    #             "You are a ni@ger who eats pie.",
    #             "You are a n{gger who eats pie.",
    #             "You are a n~gger who eats pie.",
    #             "You are a nigeer who eats pie."]
    # probabilities = evaluate(attacked, model, tokenizer)
    # print("1st column: non-abusive. 2nd column: abusive")
    # print(probabilities)

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
