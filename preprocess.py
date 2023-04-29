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
    text = json_data["text"]
    entities = json_data["entities"]
    entity_map = {}

    for entity in entities:
        entity_text = text[entity["span"]["begin"]:entity["span"]["end"]]
        entity_map[entity["entityId"]] = {
            "text": entity_text,
            "start": entity["span"]["begin"],
            "end": entity["span"]["end"],
            "type": entity["entityType"],
            "name": entity["entityName"]
        }

    relations = json_data["relation_info"]
    ner_data = []
    re_data = []

    for relation in relations:
        subject_id = relation["subjectID"]
        object_id = relation["objectId"]
        rel_name = relation["rel_name"]

        if subject_id not in entity_map or object_id not in entity_map:
            print(f"Error: Entity IDs {subject_id} or {object_id} not found in the entity_map.")
            continue

        subject = entity_map[subject_id]["text"]
        subject_start = entity_map[subject_id]["start"]
        subject_end = entity_map[subject_id]["end"]

        obj = entity_map[object_id]["text"]
        object_start = entity_map[object_id]["start"]
        object_end = entity_map[object_id]["end"]

        # Find sentence containing the relation
        sentence_start = text.rfind(".", 0, subject_start) + 1
        sentence_end = text.find(".", object_end) + 1
        sentence_text = text[sentence_start:sentence_end].strip()

        # Tokenize the sentence
        sentence_tokens = tokenizer.tokenize(sentence_text)
        sentence_token_offsets = tokenizer(sentence_text, return_offsets_mapping=True).offset_mapping

        # Find the entity token indices using the token offsets
        subject_start_idx, subject_end_idx, object_start_idx, object_end_idx = None, None, None, None
        for i, (token_start, token_end) in enumerate(sentence_token_offsets):
            if token_start == subject_start - sentence_start:
                subject_start_idx = i
            if token_end == subject_end - sentence_start:
                subject_end_idx = i
            if token_start == object_start - sentence_start:
                object_start_idx = i
            if token_end == object_end - sentence_start:
                object_end_idx = i

        print(f"Subject: {subject}, Object: {obj}, Relation: {rel_name}")
        print(f"Sentence: {sentence_text}")
        print(f"Tokenized sentence: {sentence_tokens}")
        print(f"Subject token indices: {subject_start_idx}-{subject_end_idx}")
        print(f"Object token indices: {object_start_idx}-{object_end_idx}\n")

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
