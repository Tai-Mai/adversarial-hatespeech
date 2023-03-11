import transformers

def attack(sentence, model, tokenizer):
    model = model.to(device)

