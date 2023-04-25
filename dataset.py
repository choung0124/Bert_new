import torch
from torch.utils.data import Dataset

class NERRE_Dataset(Dataset):
    def __init__(self, sentences, subject_labels, object_labels, re_labels):
        self.sentences = sentences
        self.subject_labels = subject_labels
        self.object_labels = object_labels
        self.re_labels = re_labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.subject_labels[index], self.object_labels[index], self.re_labels[index]
