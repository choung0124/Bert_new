import os
import json

entity_dict = {}
relation_dict = {}

# Directory containing JSON files
json_dir = "test"

# Loop over each JSON file in directory
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        # Load JSON file
        with open(os.path.join(json_dir, filename), "r") as f:
            data = json.load(f)
        
        # Extract entities
        entities = data["entities"]
        for entity in entities:
            entity_id = entity["entityId"]
            entity_dict[entity_id] = {
                "name": entity["entityName"],
                "type": entity["entityType"]
            }
        
        # Extract relations
        relations = data.get("relation_info", [])
        for relation in relations:
            subject_id = relation["subjectID"]
            object_id = relation["objectId"]
            relation_name = relation["rel_name"]
            if subject_id in entity_dict and object_id in entity_dict:
                if subject_id not in relation_dict:
                    relation_dict[subject_id] = {}
                if object_id not in relation_dict[subject_id]:
                    relation_dict[subject_id][object_id] = []
                relation_dict[subject_id][object_id].append(relation_name)
                
# Print entity and relation dictionaries
print("Entity dictionary:\n", entity_dict)
print("\nRelation dictionary:\n", relation_dict)
