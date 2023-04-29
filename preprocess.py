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

def preprocess_data(json_data):
    # Check and fix misplaced "relation_info" field
    if "relation_info" not in json_data and "relation_info" in json_data["entities"][-1]:
        json_data["relation_info"] = json_data["entities"][-1]["relation_info"]
        del json_data["entities"][-1]["relation_info"]

    entity_dict = {}
    relation_dict = {}
    label_to_id = {}
    relation_to_id = {}
    
    ner_data = []
    re_data = []

    # Extract entities
    entities = json_data["entities"]
    for entity in entities:
        entity_id = entity["entityId"]
        entity_dict[entity_id] = {
            "name": entity["entityName"],
            "type": entity["entityType"],
            "span": entity["span"]
        }

    # Get entities from full text
    text = json_data["text"]
    entity_spans = [entity["span"] for entity in entities]
    entity_texts = [text[span["begin"]:span["end"]] for span in entity_spans]

    # Split the text into sentences
    sentences = sent_tokenize(text)

    # Filter sentences containing entities
    relevant_sentences = []
    for sentence in sentences:
        if any(entity_text in sentence for entity_text in entity_texts):
            relevant_sentences.append(sentence)

    # Process entities
    entity_positions = {}
    for entity_id, entity in entity_dict.items():
        entity_text = text[entity["span"]["begin"]:entity["span"]["end"]]

        # Find the relevant sentence index containing the entity
        sentence_idx = next((i for i, sentence in enumerate(relevant_sentences) if entity_text in sentence), None)
        if sentence_idx is not None:
            sentence_text = relevant_sentences[sentence_idx]
            sentence_tokens = tokenizer.tokenize(sentence_text)
            sentence_token_offsets = tokenizer(sentence_text, return_offsets_mapping=True).offset_mapping

            # Find the entity token indices using the token offsets
            entity_start_idx, entity_end_idx = None, None
            for i, (token_start, token_end) in enumerate(sentence_token_offsets):
                if token_start == entity["span"]["begin"]:
                    entity_start_idx = i
                if token_end == entity["span"]["end"]:
                    entity_end_idx = i
                    break

            if entity_start_idx is None or entity_end_idx is None:
                print(f"Unable to find the entity '{entity_text}' in the sentence '{sentence_text}'")
                continue

            entity_positions[entity_id] = (sentence_idx, entity_start_idx, entity_end_idx)

        # Annotate the tokens with the entity label
            for i, token in enumerate(sentence_tokens):
                if i == entity_start_idx:
                    label = f"B-{entity['type']}-{entity['name']}"
                elif (entity_start_idx is not None and entity_end_idx is not None) and (entity_start_idx < i <= entity_end_idx):
                    label = f"I-{entity['type']}-{entity['name']}"
                else:
                    label = "O"

                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
                ner_data.append((token, label_to_id[label]))

    # Process relations
    if "relation_info" in json_data:
        relations = json_data["relation_info"]
        for relation in relations:
            relation_id = relation["relationId"]
            relation_type = relation["relationType"]
            if relation_type not in relation_to_id:
                relation_to_id[relation_type] = len(relation_to_id)
            relation_dict[relation_id] = {
                "type": relation_type,
                "entity_pair": relation["entityPair"]
            }

            # Get sentence, start, and end indices for both entities
            entity1_id, entity2_id = relation["entityPair"]
            entity1_pos = entity_positions.get(entity1_id)
            entity2_pos = entity_positions.get(entity2_id)

            if entity1_pos is None or entity2_pos is None:
                continue

            # Check if entities are in the same sentence
            if entity1_pos[0] == entity2_pos[0]:
                sentence_idx = entity1_pos[0]
                sentence = relevant_sentences[sentence_idx]
                re_data.append({
                    "tokens": tokenizer.tokenize(sentence),
                    "relation_type": relation_to_id[relation_type],
                    "entity1_pos": (entity1_pos[1], entity1_pos[2]),
                    "entity2_pos": (entity2_pos[1], entity2_pos[2])
                })

    return ner_data, re_data


# Read JSON files
for file in os.listdir(json_directory):
    if file.endswith(".json"):
        with open(os.path.join(json_directory, file), "r") as json_file:
            json_data = json.load(json_file)
            ner_data, re_data = preprocess_data(json_data)
            preprocessed_ner_data.extend(ner_data)
            preprocessed_re_data.extend(re_data)

# Save preprocessed data
with open("preprocessed_ner_data.pkl", "wb") as ner_file:
    pickle.dump(preprocessed_ner_data, ner_file)

with open("preprocessed_re_data.pkl", "wb") as re_file:
    pickle.dump(preprocessed_re_data, re_file)

print("Preprocessing completed.")
