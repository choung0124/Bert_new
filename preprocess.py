import json
import pickle
from transformers import BertTokenizerFast
import os
from nltk import sent_tokenize
import itertools
import spacy
import nltk
from fuzzywuzzy import fuzz
nlp = spacy.load("en_core_web_sm")

# Initialize the tokenizer
label_to_id = {}
relation_to_id = {}
json_directory = "test"
preprocessed_ner_data = []
preprocessed_re_data = []

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def find_best_match(entity_text, sentence_text, sentence_tokens, sentence_token_offsets):
    best_match = None
    best_score = 0
    for i, (token_start, token_end) in enumerate(sentence_token_offsets):
        token_text = sentence_text[token_start:token_end]
        score = fuzz.ratio(entity_text, token_text)
        if score > best_score:
            best_score = score
            best_match = (i, token_start, token_end)
    return best_match

# Existing preprocessing functions
def preprocess_data(json_data, tokenizer, label_to_id, relation_to_id):
    # Check and fix misplaced "relation_info" field
    if "relation_info" not in json_data and "relation_info" in json_data["entities"][-1]:
        json_data["relation_info"] = json_data["entities"][-1]["relation_info"]
        del json_data["entities"][-1]["relation_info"]

    ner_data = []
    re_data = []

    entities_dict = {entity["entityId"]: entity for entity in json_data["entities"]}
    entity_ids = set(entities_dict.keys())

    relation_dict = {}
    for relation in json_data["relation_info"]:
        subject_id = relation["subjectID"]
        obj_id = relation["objectId"]
        if subject_id not in relation_dict:
            relation_dict[subject_id] = {}
        relation_dict[subject_id][obj_id] = relation["rel_name"]

    text = json_data["text"]

    # Split the text into sentences
    sentences = sent_tokenize(text)

    sentence_boundaries = [0]
    for sentence in sentences[:-1]:
        sentence_boundaries.append(sentence_boundaries[-1] + len(sentence) + 1)

    # Filter sentences containing entities
    relevant_sentences = []
    for i, (sentence, boundary) in enumerate(zip(sentences, sentence_boundaries)):
        if any(boundary <= entity["span"]["begin"] < boundary + len(sentence) for entity in json_data["entities"]):
            relevant_sentences.append((sentence, i, boundary))

    # Process entities
    entity_positions = {}
    for entity in sorted(json_data["entities"], key=lambda x: x["span"]["begin"]):
        begin, end = entity["span"]["begin"], entity["span"]["end"]
        entity_text = json_data["text"][begin:end]

        # Find the relevant sentence index containing the entity
        sentence_idx_boundary = next(((i, boundary) for sentence, i, boundary in relevant_sentences if boundary + len(sentence) >= begin), (None, None))
        if sentence_idx_boundary[0] is not None:
            sentence_idx, boundary = sentence_idx_boundary
        else:
            print(f"Entity '{entity_text}' not found in any relevant sentence. Entity data: {entity}")
            continue  # Skip the current entity if no relevant sentence is found

        sentence_text = relevant_sentences[sentence_idx][0]
        sentence_tokens = tokenizer.tokenize(sentence_text)
        sentence_token_offsets = tokenizer(sentence_text, return_offsets_mapping=True).offset_mapping

        # Find the first occurrence of the entity text in the sentence text
        entity_start = sentence_text.find(entity_text)
        if entity_start != -1:
            entity_end = entity_start + len(entity_text) - 1
        else:
            print(f"Unable to find the entity '{entity_text}' in the sentence '{sentence_text}'")
            continue

        entity_positions[entity["entityId"]] = (sentence_idx, entity_start, entity_end)

        # Tokenize the relevant sentence
        sentence_text = relevant_sentences[sentence_idx][0]
        sentence_tokens = tokenizer.tokenize(sentence_text)
        sentence_token_offsets = tokenizer(sentence_text, return_offsets_mapping=True).offset_mapping

        # Tokenize the entity text
        entity_text = text[begin:end]
        entity_tokens = tokenizer.tokenize(entity_text)

        # Find the first occurrence of the entity tokens in the sentence tokens
        best_match = find_best_match(entity_text, sentence_text, sentence_tokens, sentence_token_offsets)
        if best_match:
            entity_start_idx, begin, end = best_match
            entity_end_idx = entity_start_idx + len(entity_tokens) - 1
        else:
            raise ValueError(f"Unable to find the entity '{entity_text}' in the sentence '{sentence_text}'")

        # Annotate the tokens with the entity label
        for i, token in enumerate(sentence_tokens):
            if i == entity_start_idx:
                label = f"B-{entity['entityType']}-{entity['entityName']}"
            elif (entity_start_idx is not None and entity_end_idx is not None) and (entity_start_idx < i <= entity_end_idx):
                label = f"I-{entity['entityType']}-{entity['entityName']}"
            else:
                label = "O"

            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)

            ner_data.append((token, label, len(ner_data)))

        if f"{entity['entityType']}-{entity['entityName']}" not in label_to_id:
            label_to_id[f"{entity['entityType']}-{entity['entityName']}"] = len(label_to_id)

    # Process relations
    for relation in json_data["relation_info"]:
        subject_id = relation["subjectID"]
        obj_id = relation["objectId"]
        rel_name = relation["rel_name"]

        if subject_id not in entity_positions or obj_id not in entity_positions:
            print(f"Skipping relation between '{subject_id}' and '{obj_id}' as one or both entities were not found in any relevant sentence.")
            continue

        subject_sentence_idx, subject_start, subject_end = entity_positions[subject_id]
        object_sentence_idx, object_start, object_end = entity_positions[obj_id]

        if subject_sentence_idx != object_sentence_idx:
            print(f"Skipping relation between '{subject_id}' and '{obj_id}' as they are not in the same sentence.")
            continue

        sentence_text = relevant_sentences[subject_sentence_idx][0]

        re_data.append({
            'id': (subject_id, obj_id),
            'subject': sentence_text[subject_start:subject_end + 1],
            'object': sentence_text[object_start:object_end + 1],
            'relation': rel_name,
            'subject_tokens': tokenizer.tokenize(sentence_text[subject_start:subject_end + 1]),
            'object_tokens': tokenizer.tokenize(sentence_text[object_start:object_end + 1])
        })

        if rel_name not in relation_to_id:
            relation_to_id[rel_name] = len(relation_to_id)

    return ner_data, re_data, label_to_id, relation_to_id

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
                try:
                    preprocessed_file_data = preprocess_data(json_data, tokenizer, label_to_id, relation_to_id)
                    preprocessed_data.extend(preprocessed_file_data)
                except ValueError as e:
                    print(f"Error processing {json_path}: {e}")
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
    
# Save the label dictionaries to disk
with open("label_to_id.pkl", "wb") as f:
    pickle.dump(label_to_id, f)

with open("relation_to_id.pkl", "wb") as f:
    pickle.dump(relation_to_id, f)
