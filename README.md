# GoEmotions for Portuguese

This repository contains scripts for downloading, translating the datasets and fine-tuning a bert model for portuguese
emotion classification based on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset.
Original and Ekman taxonomy are supported.

## Requirements

Create an environment, clone this repository and run:
```
pip install -r requirements.txt
```

## Dataset

First you should download and translate the original dataset, for this run the following script:
```
python translate.py
```
When the script finishes all the data needed should be on the dataset folder.

## Fine-tuning

To perform the fine-tuning you should run:
```
python run_training.py \
--model_output_dir fine_tuned_model
```
You can change the folder where you save the model by passing a folder name in the --model_output_dir argument

There are optional arguments you can pass:

--batch_size\
Batch size for training (default = 16)\
--max_seq_length\
Maximum sequence length (default = 128)\
--model_name\
Name of the model to be used (default = "neuralmind/bert-base-portuguese-cased")\
--n_epochs\
Number of epochs to run fine tuning (default = 4)\
--warmup_proportion\
Float number between 0 and 1 that represents the proportion for warmup (default = 0.2)\
--beta\
Beta parameter for weighing method Class-Balanced Loss (default = 0.999)\
--no_cuda\
If passed True, gpu will not be used (default = False)\
--seed\
Seed for pseudo-random number generation for pytorch, numpy, python.random (default = 42)\
--taxonomy\
Select which taxonomy to be used, original or ekman (default = "original")\
--resume_from_checkpoint\
If passed, should be a path for a checkpoint file (default = None)

# Using Fine-tuned Model
```python
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from pprint import pprint

model_path = 'fine_tuned_model'

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)


classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)

threshold = 0.3

inputs = [
	'Eu te amo',
	'Eu acho que você é uma ótima pessoa',
	'Eu odeio aquele cara',
	]

output = classifier(inputs)

predictions = []

for prediction in output:
	predictions.append(list(x for x in prediction if x['score']>= threshold))

pprint(predictions)

# Output
# [[{'label': 'amor', 'score': 0.9658263325691223}],
#  [{'label': 'admiração', 'score': 0.9569578170776367}],
#  [{'label': 'raiva', 'score': 0.6997460126876831}]]
```
