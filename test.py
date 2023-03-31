from transformers import AutoTokenizer #, AutoModelForSequenceClassification
from pretrained.models import *
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
        "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
)

text = "why are you repeating yourself are you a little retarded"

tokenized = tokenizer(text)#.to(device)

print(tokenized)
