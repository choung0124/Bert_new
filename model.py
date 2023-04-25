import torch
from torch import nn
from transformers import BertModel

class NER_RE_Model(nn.Module):
    def __init__(self, ner_dim, re_dim):
        super(NER_RE_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.subject_ner_classifier = nn.Linear(768, ner_dim)  # Separate classifier for subject NER
        self.object_ner_classifier = nn.Linear(768, ner_dim)  # Separate classifier for object NER
        self.re_classifier = nn.Linear(768, re_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_output[0]

        batch_size = input_ids.shape[0]

        ner_logits_subject = self.subject_ner_classifier(sequence_output).view(batch_size, -1, self.subject_ner_classifier.out_features)[:, 0]
        ner_logits_object = self.object_ner_classifier(sequence_output).view(batch_size, -1, self.object_ner_classifier.out_features)[:, 0]
        re_logits = self.re_classifier(sequence_output).view(batch_size, -1, self.re_classifier.out_features)[:, 0]

        return ner_logits_subject, ner_logits_object, re_logits



