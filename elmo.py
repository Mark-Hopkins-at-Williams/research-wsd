# -*- coding: utf-8 -*-
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

N_REPRESENTATIONS = 3
elmo = Elmo(options_file, weight_file, N_REPRESENTATIONS, dropout=0.05)
def elmo_vectorize(instance):
    tokens = instance.tokens
    sense = instance.sense
    position = instance.pos

    elmo_ids = batch_to_ids([tokens])
    embeddings = elmo(elmo_ids)

    representation_summed = sum(embeddings).squeeze(0)

    return representation_summed[position].detach()

