import torch
from torch.utils.data import Dataset

class NERRE_Dataset(Dataset):
    def __init__(self, sentences, ner_labels, re_labels):
        self.sentences = sentences
        self.ner_labels = ner_labels
        self.re_labels = re_labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.ner_labels[idx], self.re_labels[idx]
