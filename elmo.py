# -*- coding: utf-8 -*-
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_transformers import BertTokenizer
from util import cudaify
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

N_REPRESENTATIONS = 3
elmo = Elmo(options_file, weight_file, N_REPRESENTATIONS)
elmo = cudaify(elmo)

def elmo_vectorize(positions, vectors):
    print("vectorizing")
    elmo_ids = batch_to_ids(vectors)
    elmo_ids = cudaify(elmo_ids)
    outputs_dict = elmo(elmo_ids)
    positions = cudaify(torch.tensor(positions))

    representations = outputs_dict["elmo_representations"]
    representations_avged = sum(representations) / 2
    
    gather_index = torch.stack([positions] * representations_avged.shape[2], dim=1).unsqueeze(1)

    positioned = representations_avged.gather(1, gather_index).squeeze(1).cpu()

    return positioned.detach()


