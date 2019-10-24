import torch

from transformers import BertModel, BertTokenizer

# Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text
def vectorize_instance(instance):
    input_ids = torch.tensor([tokenizer.encode(' '.join(instance.tokens), 
                                               add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        last_hidden_states = last_hidden_states.squeeze(0)
        return last_hidden_states[instance.pos]

""" 
This code appears to be broken, specifically it behaves nondeterministically. 

"""
"""
config = BertConfig.from_pretrained('bert-base-uncased')
config.output_hidden_states = True

def generate_vectorization_broken(layers_i, add_sent):
    def vectorize_instance(instance):        
        tokens = instance.tokens
        position = instance.pos
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
        with torch.no_grad():
            results = bert(input_ids)
            hidden_layers = results[2]
            total = torch.zeros(hidden_layers[0].squeeze(0)[position].shape)
            for i in layers_i:
                curr_vec = hidden_layers[i].squeeze(0)[position]
                total += curr_vec
            if add_sent:
                total += results[1].squeeze(0)
                avg_vec = total / (len(layers_i) + 1)
            else:
                avg_vec = total / len(layers_i)
            return avg_vec.detach()

    return vectorize_instance

vectorize_instance = generate_vectorization([12], False)
"""