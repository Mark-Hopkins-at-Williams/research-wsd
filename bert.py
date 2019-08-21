import torch

from pytorch_transformers import BertModel, BertConfig, BertTokenizer

config = BertConfig.from_pretrained('bert-base-uncased')
config.output_hidden_states = True
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel(config)


def generate_vectorization(layers_i, add_sent):
    def vectorize_instance(instance):
        """
        Converts a SenseInstance into a tensor.
        
        """
        tokens = instance.tokens
        tokens = ["CLS"] + tokens + ["SEP"]
        position = instance.pos + 1
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
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
