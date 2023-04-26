import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from BERT_full_train import BertForNERAndRE
import json
import sys
import os

model_path = "models/combined/" # Change this to the path of your pretrained model directory

# Load the label_to_id and relation_to_id mappings from the training data
with open(os.path.join(model_path, "label_to_id.json"), "r") as f:
    label_to_id = json.load(f)

with open(os.path.join(model_path, "relation_to_id.json"), "r") as f:
    relation_to_id = json.load(f)
    
# Invert the label_to_id dictionary to get the id_to_label mapping
id_to_label = {v: k for k, v in label_to_id.items()}

# Create a list of NER labels in the correct order
ner_labels = [id_to_label[i] for i in range(len(label_to_id))]

# Load the fine-tuned model with the correct number of labels
config = BertConfig.from_pretrained(os.path.join(model_path, "config.json"))
model = BertForNERAndRE.from_pretrained(model_path, config=config, num_ner_labels=len(label_to_id), num_re_labels=len(relation_to_id))
tokenizer = BertTokenizer.from_pretrained(model_path)

# Set the device (CPU or GPU) to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to extract relationships from input text
def extract_relationships(text):
    # Split the input text into chunks of length max_length
    chunks = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i+max_length]
        chunks.append(chunk)

    # Tokenize and align the NER labels for each chunk
    input_ids_list = []
    attention_masks_list = []
    for chunk in chunks:
        tokens = tokenizer.encode_plus(chunk, add_special_tokens=True, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        input_ids_list.append(tokens["input_ids"])
        attention_masks_list.append(tokens["attention_mask"])

    # Run the model on each chunk separately and concatenate the outputs
    relationships = []
    ner_labels = []
    for i in range(0, len(chunks), batch_size):
        input_ids = torch.cat(input_ids_list[i:i+batch_size], dim=0)
        attention_masks = torch.cat(attention_masks_list[i:i+batch_size], dim=0)
        with torch.no_grad():
            outputs = model(input_ids.to(device), attention_mask=attention_masks.to(device))
        ner_logits = outputs[0]
        ner_label_ids = ner_logits.argmax(dim=2)
        for j in range(len(input_ids)):
            ner_tokens = tokenizer.convert_ids_to_tokens(input_ids[j])
            ner_label = []
            for label_id in ner_label_ids[j].tolist():
                ner_label.append(label_list[label_id])
            ner_labels.append(ner_label)
    entities = extract_entities(ner_tokens, ner_labels)
    if len(entities) >= 2:
        # Extract relations between entities
        for i in range(len(entities)-1):
            for j in range(i+1, len(entities)):
                relation = extract_relation(text, entities[i], entities[j])
                if relation:
                    relationships.append((entities[i], relation, entities[j]))
    return relationships


# Get the input text from the command line argument
text = sys.argv[1]

# Extract the relationships from the input text
relationships = extract_relationships(text)

# Print the extracted relationships
print("Extracted relationships:")
for subject, relationship, obj in relationships:
    print(f"{subject} - {relationship} - {obj}")
