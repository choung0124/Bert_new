import torch
from transformers import BertTokenizer, BertForNERAndRE

# Load the tokenizer and model from the output directory
output_dir = "models/combined"
tokenizer = BertTokenizer.from_pretrained(output_dir)
model = BertForNERAndRE.from_pretrained(output_dir)

# Set the model to evaluation mode
model.eval()

# Define a function to predict NER and RE labels
def predict_labels(text):
    # Tokenize the input text
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

    # Predict NER labels
    ner_outputs = model(input_ids)[0]
    ner_labels = torch.argmax(ner_outputs, axis=-1).squeeze(0)
    ner_labels = [tokenizer.convert_ids_to_tokens(token_id) for token_id in ner_labels]

    # Predict RE labels
    re_outputs = model(input_ids)[1]
    re_labels = torch.argmax(re_outputs, axis=-1).squeeze(0)
    re_labels = tokenizer.convert_ids_to_tokens(re_labels)

    return ner_labels, re_labels
