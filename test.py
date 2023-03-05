import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
### from models.py
from hatexplain.models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two")
model = \
    Model_Rational_Label.from_pretrained(
        "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two"
    )
model = model.to(device)
inputs = tokenizer('He is a great guy', return_tensors="pt").to(device)
prediction_logits, _ = model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
softmax = torch.nn.Softmax(dim=1)
probs = softmax(prediction_logits)
print(f"Normal: {probs[0][0]}\nHatespeech: {probs[0][1]}")
