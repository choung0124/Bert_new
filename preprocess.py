import json
import pickle
from DistiliBERT_train import preprocess_data
from transformers import DistilBertTokenizerFast
import os

# Load the input data from a JSON file
with open("input_data.json", "r") as f:
    json_data_list = json.load(f)

# Initialize the tokenizer
label_to_id = {}
relation_to_id = {}
json_directory = "test"
preprocessed_ner_data = []
preprocessed_re_data = []

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Iterate through all JSON files in the directory
def validate_json(json_data):
    # Check if the necessary keys are present
    if "entities" not in json_data or "relation_info" not in json_data or "text" not in json_data:
        return False

    # Check if there are entities and relations
    if len(json_data["entities"]) == 0 or len(json_data["relation_info"]) == 0:
        return False

    # Additional validation criteria can be added here based on your data format

    return True

for file_name in os.listdir(json_directory):
    if file_name.endswith(".json"):
        json_path = os.path.join(json_directory, file_name)

        try:
            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)

            if validate_json(json_data):
                preprocessed_file_data = preprocess_data(json_data, tokenizer, label_to_id, relation_to_id)
                preprocessed_data.extend(preprocessed_file_data)
            else:
                print(f"Skipping {json_path} due to invalid JSON data")
        except json.JSONDecodeError as e:
            print(f"Error loading {json_path}: {e}")
            continue

for json_data in json_data_list:
    ner_data, re_data, label_to_id, relation_to_id = preprocess_data(json_data, tokenizer, label_to_id, relation_to_id)
    preprocessed_ner_data.append(ner_data)
    preprocessed_re_data.append(re_data)

# Save the preprocessed data to disk
with open("preprocessed_ner_data.pkl", "wb") as f:
    pickle.dump(preprocessed_ner_data, f)

with open("preprocessed_re_data.pkl", "wb") as f:
    pickle.dump(preprocessed_re_data, f)
