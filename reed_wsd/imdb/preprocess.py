import torch
from transformers import BertModel, BertTokenizer
from reed_wsd.util import cudaify
from tqdm import tqdm
import os.path as path
import os
import json

file_dir = path.dirname(path.realpath(__file__))
imdb_path = path.join(file_dir, 'data/aclImdb/')
DEVICE = torch.device('cuda:2')

def sequence_to_vec(sequence, tokenizer, bert, max_seq_length=512):
    tknz_output = tokenizer(sequence,
                            add_special_tokens=True, 
                            return_tensors='pt',
                            verbose=False)
    if tknz_output['input_ids'].shape[1] >= max_seq_length:
        return None
    tknz_output = tknz_output.to(DEVICE)
    with torch.no_grad():
        last_hidden_states = bert(**tknz_output)[0].squeeze(0)
    cls_embedding = last_hidden_states[0]
    return cls_embedding

class IMDBPreprocessor:
    def __init__(self):
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_path = path.join(file_dir, 'data/aclImdb/train')
        self.test_path = path.join(file_dir, 'data/aclImdb/test')

    def vectorize_file(self, file_path, positive):
        with open(file_path, 'r') as f:
            comment = f.read()
        comment_vec = sequence_to_vec(comment,
                                      self.tokenizer,
                                      self.bert)
        if comment_vec is not None:
            return {'vec': comment_vec.cpu().numpy().tolist(),
                    'gold': int(positive)}
    
    def vectorize_folder(self, folder_path, positive):
        instances = []
        file_names = os.listdir(folder_path)
        for f_name in tqdm(file_names):
            inst = self.vectorize_file(path.join(folder_path, f_name), positive)
            if inst is not None: 
                instances.append(inst)
        return instances

    def __call__(self):
        train_set = (self.vectorize_folder(path.join(self.train_path, 'pos'), True) + 
                    self.vectorize_folder(path.join(self.train_path, 'neg'), False))
        test_set = (self.vectorize_folder(path.join(self.test_path, 'pos'), True) +
                   self.vectorize_folder(path.join(self.test_path, 'neg'), False))
        with open(path.join(imdb_path, 'imdb.json'), 'w') as f:
            json.dump({'train': train_set, 'test': test_set}, f)              

if __name__ == '__main__':
    pp = IMDBPreprocessor()
    pp()
