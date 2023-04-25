import torch
from torch.utils.data import Dataset

class NERRE_Dataset(Dataset):
    def __init__(self, dir_path):
        full_text, full_entities, full_relations = read_json_files(dir_path)
        entity_sentences, relation_sentences = extract_sentences(full_text, full_entities, full_relations)
        self.sentences = [item['sentence'] for item in entity_sentences + relation_sentences]
        self.subject_labels = [None] * len(entity_sentences) + [item['subject'] for item in relation_sentences]
        self.object_labels = [None] * len(entity_sentences) + [item['object'] for item in relation_sentences]
        self.re_labels = [None] * len(entity_sentences) + [item['relation'] for item in relation_sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.subject_labels[idx], self.object_labels[idx], self.re_labels[idx]
