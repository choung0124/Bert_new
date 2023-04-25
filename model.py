import torch
from torch import nn
from transformers import BertModel

class NER_RE_Model(nn.Module):
    def __init__(self, ner_dim, re_dim):
        super(NER_RE_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.subject_ner_classifier = nn.Linear(768, ner_dim)  # Separate classifier for subject NER
        self.object_ner_classifier = nn.Linear(768, ner_dim)  # Separate classifier for object NER
        self.regular_ner_classifier = nn.Linear(768, ner_dim)  # Additional classifier for regular NER
        self.re_classifier = nn.Linear(768, re_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token_output = bert_output.last_hidden_state[:, 0, :]
        ner_logits_subject = self.subject_ner_classifier(cls_token_output)
        ner_logits_object = self.object_ner_classifier(cls_token_output)
        ner_logits_regular = self.regular_ner_classifier(cls_token_output)  # Get logits for regular NER
        re_logits = self.re_classifier(cls_token_output)
        return ner_logits_subject, ner_logits_object, ner_logits_regular, re_logits  # Return regular NER logits too
