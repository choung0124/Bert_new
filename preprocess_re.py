import json
import os

def preprocess_re(json_data):
    re_data = []
    entities = {entity["entityId"]: entity for entity in json_data["entities"]}

    for relation in json_data["relation_info"]:
        subject_id, obj_id = relation["subjectID"], relation["objectId"]
        if subject_id not in entities or obj_id not in entities:
            continue
        subject = entities[subject_id]["entityName"]
        obj = entities[obj_id]["entityName"]
        re_data.append((subject, relation["rel_name"], obj))

    return re_data


# Set the directory containing the JSON files
json_data_dir = "test"

# Preprocessed data
preprocessed_data = []

# Iterate through all JSON files in the directory
for file_name in os.listdir(json_data_dir):
    if file_name.endswith(".json"):
        json_path = os.path.join(json_data_dir, file_name)

        # Load the JSON data
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        # Preprocess the data for RE tasks
        re_data = preprocess_re(json_data)
        preprocessed_data.append(re_data)
        print(f"Processed: {file_name}")
