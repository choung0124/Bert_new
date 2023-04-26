import re

# Define a regular expression pattern to match entity IDs in brackets, e.g. [e1]
entity_id_pattern = re.compile(r"\[e\d+\]")

def extract_entities(text):
    entities = {}
    # Find all matches of entity ID pattern
    for entity_id in entity_id_pattern.findall(text):
        # Remove brackets from entity ID
        entity_id = entity_id[1:-1]
        # Replace entity ID in text with a placeholder
        text = text.replace(entity_id_pattern.sub(f"@{entity_id}@", entity_id), "")
        # Extract entity type and name from entity ID
        entity_type, entity_name = entity_id.split("_")
        entities[entity_id] = {"type": entity_type, "name": entity_name}
    return text, entities

def extract_relations(text, entities):
    relations = []
    # Split text into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    for sentence in sentences:
        # Find all matches of relation pattern
        for match in relation_pattern.finditer(sentence):
            # Extract subject and object entities using entity IDs
            subject_id = match.group("subject")
            object_id = match.group("object")
            # Ignore relations with unknown entities
            if subject_id not in entities or object_id not in entities:
                continue
            # Extract relation name and add relation to list
            relation_name = match.group("relation")
            relation = {"subject": entities[subject_id], "object": entities[object_id], "name": relation_name}
            relations.append(relation)
    return relations

