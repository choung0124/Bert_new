import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import sys


# Load the pretrained model and tokenizer
model_path = "models/combined/" # Change this to the path of your pretrained model directory
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set the device (CPU or GPU) to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to extract relationships from input text
def extract_relationships(text, model_path, max_length=512, batch_size=8):
    # Load the pretrained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForNERAndRE.from_pretrained(model_path)
    
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
    for i in range(0, len(chunks), batch_size):
        input_ids = torch.cat(input_ids_list[i:i+batch_size], dim=0)
        attention_masks = torch.cat(attention_masks_list[i:i+batch_size], dim=0)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks)
        ner_logits = outputs[0]
        ner_label_ids = ner_logits.argmax(dim=2)
        for j in range(len(input_ids)):
            ner_tokens = tokenizer.convert_ids_to_tokens(input_ids[j])
            ner_labels = [label_list[label_id] for label_id in ner_label_ids[j].tolist()]
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
relationships = extract_relationships(text, "models/combined")

# Print the extracted relationships
print("Extracted relationships:")
for subject, relationship, obj in relationships:
    print(f"{subject} - {relationship} - {obj}")
