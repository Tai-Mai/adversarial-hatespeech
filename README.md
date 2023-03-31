# adversarial-hatespeech

## Installation
```bash
$ pip install transformers nltk lime pipreqs
$ pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
* `nltk` for detokenizing
* `lime` for explaining
* `pipreqs` for creating `requirements.txt`

Alternatively: (not tested)
```bash
$ pip install -r requirements.txt
```

## Usage
* All important functions are documented with docstrings
```bash
$ git clone https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two
```

### Preparation
* Clone the huggingface repo of the HateXplain model

### 1. Find adversarial examples
* Run `batchscripts/attack.sh`
* Resulting adversarial examples are saved in `data/attacks_val_no-letters.json`
    (already done in this repo)

### 2. Analyze adversarial examples for stats
* Run `batchscripts/analyze.sh`
* Results are printed to terminal/saved in `outputs/analyze_val_no-letters.txt`
    (already done in this repo)

#### 2.5 (Optional) Explain adversarial examples with LIME
* Run `batchscripts/explain.sh`
* Explanations are saved into the existing `data/attacks_val_no-letters.json`

### 3. Test adversarial examples on test split
* Run `batchscripts/test.sh`
* Unsuccessful attacks are saved in `data/test_val_no-letters_unsuccessful.json`
    (already done in this repo)
