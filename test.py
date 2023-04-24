import torch
import argparse
from transformers import BertTokenizer
from train import MyModel, idx2ner_label, idx2re_label

# Set up the argument parser
parser = argparse.ArgumentParser(description="Test the model with a random text.")
parser.add_argument("input_text", type=str, help="The input text to be labeled.")
args = parser.parse_args()

# Load the trained model
model = MyModel()
model.load_state_dict(torch.load("trained_model.pt"))
model.eval()

# Tokenize the input text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(args.input_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']
token_type_ids = tokens['token_type_ids']

# Pass the tokenized input to the model
with torch.no_grad():
    ner_logits, re_logits = model(input_ids, attention_mask, token_type_ids)

# Decode the output logits to get the predicted labels
ner_predictions = torch.argmax(ner_logits, dim=-1)
re_predictions = torch.argmax(re_logits, dim=-1)

# Convert the predicted label indices to their corresponding label names
ner_label = idx2ner_label[ner_predictions.item()]
re_label = idx2re_label[re_predictions.item()]

print(f"Predicted NER label: {ner_label}")
print(f"Predicted RE label: {re_label}")
