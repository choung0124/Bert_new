import torch
from torch import nn
from transformers import BertModel

class NER_RE_Model(nn.Module):
    def __init__(self, ner_label_count, re_label_count):
        super(NER_RE_Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, ner_label_count)
        self.re_classifier = nn.Linear(self.bert.config.hidden_size, re_label_count)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token_output = bert_output.last_hidden_state[:, 0, :]
        ner_logits = self.ner_classifier(cls_token_output)
        re_logits = self.re_classifier(cls_token_output)
        return ner_logits, re_logits
