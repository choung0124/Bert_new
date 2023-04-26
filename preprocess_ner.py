import json
import os

def preprocess_ner(json_data):
    ner_data = []
    
    for entity in json_data["entities"]:
        begin = entity["span"]["begin"]
        end = entity["span"]["end"]
        entity_type = entity["entityType"]
        entity_text = json_data["text"][begin:end]
        ner_data.append((begin, end, entity_type, entity_text))
    
    ner_data.sort(key=lambda x: x[0])
    
    text = json_data["text"]
    ner_tags = []
    current_idx = 0
    
    for begin, end, entity_type, entity_text in ner_data:
        while current_idx < begin:
            ner_tags.append((text[current_idx], "O"))
            current_idx += 1
        
        entity_tokens = tokenizer.tokenize(entity_text)
        for i in range(len(entity_tokens)):
            ner_tags.append((entity_tokens[i], f"{entity_type}"))
            current_idx += 1
    
        current_idx = end
    
    while current_idx < len(text):
        ner_tags.append((text[current_idx], "O"))
        current_idx += 1
    
    return ner_tags


# Set the directory containing the JSON files
json_directory = "test"

# Preprocessed data
preprocessed_data = []

# Iterate through all JSON files in the directory
for file_name in os.listdir(json_directory):
    if file_name.endswith(".json"):
        json_path = os.path.join(json_directory, file_name)

        # Load the JSON data
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        # Preprocess the data for NER tasks
        ner_data = preprocess_ner(json_data)
        preprocessed_data.append(ner_data)
        print(f"Processed: {file_name}")
        print(f"Number of entities: {len(json_data['entities'])}")
        for entity in json_data['entities']:
            print(entity)
