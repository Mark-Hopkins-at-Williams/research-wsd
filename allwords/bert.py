import torch

from transformers import BertModel, BertTokenizer

class BertSentenceVectorizer:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def __call__(self, sent):
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens 
        # in the right way for each model.
        input_ids = self.tokenizer.encode(sent, add_special_tokens=True) 
        bert_toks = self.tokenizer.convert_ids_to_tokens(input_ids)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
            last_hidden_states = last_hidden_states.squeeze(0)
            return bert_toks, last_hidden_states
        
    def dim(self):
        return 768
    
    
class BertVectorizer:
    
    def __init__(self):
        self.vectorizer = BertSentenceVectorizer()
       
    def __call__(self, instance):
        bert_toks, states = self.vectorizer(' '.join(instance.tokens))
        return states[instance.pos]

    def dim(self):
        return 768