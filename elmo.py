# -*- coding: utf-8 -*-
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_transformers import BertTokenizer
from util import cudaify
import torch
from IPython.core.debugger import set_trace

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
    representations_avged = sum(representations) / 3
    
    gather_index = torch.stack([positions] * representations_avged.shape[2], dim=1).unsqueeze(1)

    positioned = representations_avged.gather(1, gather_index).squeeze(1).cpu()

    return positioned.detach()


def elmo_vectorize_avg_both(positions, vectors):
    print("vectorizing")
    elmo_ids = batch_to_ids(vectors)
    elmo_ids = cudaify(elmo_ids)
    outputs_dict = elmo(elmo_ids)
    positions = cudaify(torch.tensor(positions))
    prev_positions = (positions - 1).clamp(0, 510)
    next_positions = (positions + 1).clamp(0, 510)

    representations = outputs_dict["elmo_representations"]
    representations_avged = sum(representations) / 3
    representations_avged = representations_avged
    
    gather_index = torch.cat([torch.stack([prev_positions, positions, next_positions], dim=1).unsqueeze(2)] * 1024, dim=2)
    gathered = representations_avged.gather(1, gather_index).mean(1).cpu()

    return gathered.detach()

def elmo_vectorize_avg_left(positions, vectors):
    print("vectorizing")
    elmo_ids = batch_to_ids(vectors)
    elmo_ids = cudaify(elmo_ids)
    outputs_dict = elmo(elmo_ids)
    positions = cudaify(torch.tensor(positions))
    prev_positions = (positions - 1).clamp(0, 510)

    representations = outputs_dict["elmo_representations"]
    representations_avged = sum(representations) / 3
    
    gather_index = torch.cat([torch.stack([prev_positions, positions], dim=1).unsqueeze(2)] * 1024, dim=2)
    
    positioned= representations_avged.gather(1, gather_index).mean(1).cpu()

    return positioned.detach()

def elmo_vectorize_avg_right(positions, vectors):
    print("vectorizing")
    elmo_ids = batch_to_ids(vectors)
    elmo_ids = cudaify(elmo_ids)
    outputs_dict = elmo(elmo_ids)
    positions = cudaify(torch.tensor(positions))
    next_positions = (positions + 1).clamp(0, 510)

    representations = outputs_dict["elmo_representations"]
    representations_avged = sum(representations) / 3
    
    gather_index = torch.cat([torch.stack([positions, next_positions], dim=1).unsqueeze(2)] * 1024, dim=2)
    
    positioned= representations_avged.gather(1, gather_index).mean(1).cpu()

    return positioned.detach()

def elmo_vectorize_concat_both(positions, vectors):
    print("vectorizing")
    elmo_ids = batch_to_ids(vectors)
    elmo_ids = cudaify(elmo_ids)
    outputs_dict = elmo(elmo_ids)
    positions = cudaify(torch.tensor(positions))
    prev_positions = (positions - 1).clamp(0, 510)
    next_positions = (positions + 1).clamp(0, 510)

    representations = outputs_dict["elmo_representations"]
    representations_avged = sum(representations) / 3
    
    gather_index = torch.cat([torch.stack([prev_positions, positions, next_positions], dim=1).unsqueeze(2)] * 1024, dim=2)
    
    positioned= representations_avged.gather(1, gather_index)
    positioned = positioned.reshape([positioned.shape[0], positioned.shape[1] * positioned.shape[2]]).cpu()

    return positioned.detach()

def elmo_vectorize_concat_left(positions, vectors):
    print("vectorizing")
    elmo_ids = batch_to_ids(vectors)
    elmo_ids = cudaify(elmo_ids)
    outputs_dict = elmo(elmo_ids)
    positions = cudaify(torch.tensor(positions))
    prev_positions = (positions - 1).clamp(0, 510)

    representations = outputs_dict["elmo_representations"]
    representations_avged = sum(representations) / 3
    
    gather_index = torch.cat([torch.stack([prev_positions, positions], dim=1).unsqueeze(2)] * 1024, dim=2)
    
    positioned= representations_avged.gather(1, gather_index)
    positioned = positioned.reshape([positioned.shape[0], positioned.shape[1] * positioned.shape[2]]).cpu()

    return positioned.detach()

def elmo_vectorize_concat_right(positions, vectors):
    print("vectorizing")
    elmo_ids = batch_to_ids(vectors)
    elmo_ids = cudaify(elmo_ids)
    outputs_dict = elmo(elmo_ids)
    positions = cudaify(torch.tensor(positions))
    next_positions = (positions + 1).clamp(0, 510)

    representations = outputs_dict["elmo_representations"]
    representations_avged = sum(representations) / 3
    
    gather_index = torch.cat([torch.stack([positions, next_positions], dim=1).unsqueeze(2)] * 1024, dim=2)
    
    positioned= representations_avged.gather(1, gather_index)
    positioned = positioned.reshape([positioned.shape[0], positioned.shape[1] * positioned.shape[2]]).cpu()

    return positioned.detach()


