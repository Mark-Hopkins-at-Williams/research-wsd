"""
Conversion tools from Raganato XML to a simpler JSON format.

Example usage:

python raganato.py datadir raganato.json

"""

from xml.dom import minidom
import json
from os.path import join
import sys
from collections import defaultdict

def create_sense_inventory(goldfiles):
    inventory = defaultdict(dict)
    for i, goldfile in enumerate(goldfiles):
        with open(goldfile) as reader:
            for line in reader:                
                fields = line.split()
                for orig_sense in fields[1:]:
                    (lemma, _) = orig_sense.split("%")
                    lemma_senses = inventory[lemma]
                    if orig_sense not in lemma_senses:
                        lemma_senses[orig_sense] = 1
                    else:
                        lemma_senses[orig_sense] += 1
    inventory = dict(inventory)
    result = dict()
    for lemma in inventory:
        sense_count = inventory[lemma]
        lemma_freq = [(sense_count[sense], sense) for sense in inventory[lemma]]
        lemma_freq = reversed(sorted(lemma_freq))
        result[lemma] = [item[1] for item in lemma_freq]                
    return result


def parse_raganato_gold(goldfile, inventory):
    result = dict()
    with open(goldfile) as reader:
        for line in reader:
            fields = line.split()
            instance_id = fields[0]
            sense1 = fields[1] # there are sometimes other senses; what to do?
            result[instance_id] = sense1
    return result


def parse_raganato_xml_sent(sent_node, sense_map):
    tokens = []
    for node in sent_node.childNodes:
        if node.nodeName == "wf":
            word = node.childNodes[0].nodeValue
            lemma = node.getAttribute('lemma')
            tag = node.getAttribute('pos')
            sense = None
            tokens.append(Token(word,tag,None,lemma,sense).to_dict())
        elif node.nodeName == "instance":
            word = node.childNodes[0].nodeValue
            lemma = node.getAttribute('lemma')
            tag = node.getAttribute('pos')
            sent_id = node.getAttribute('id')
            sense = sense_map[sent_id]
            tokens.append(Token(word,tag,sent_id,lemma,sense).to_dict())
        elif node.nodeName == "#text" and node.nodeValue == '\n':
            pass
        else:
            print('Warning: element {} not recognized!'.format(node.nodeName))
    result = {'sentid': sent_node.getAttribute('id'), 'words': tokens}
    return result           


def parse_raganato_xml(xmldoc, sense_map):
    corpora = xmldoc.getElementsByTagName('corpus')
    output_sents = []
    n_insts = 0
    for corpus in corpora:
        for node in corpus.childNodes:
            if node.nodeName == "text":
                for sent_node in node.childNodes:
                    if sent_node.nodeName == "sentence":
                        sent = parse_raganato_xml_sent(sent_node, sense_map) 
                        for w in sent['words']:
                            if 'sense' in w.keys(): n_insts += 1
                        sent['sentid'] = len(output_sents)
                        output_sents.append(sent)
                    elif sent_node.nodeName == "#text" and sent_node.nodeValue == '\n':
                        pass
                    else:
                        print('Warning: element {} not recognized!'.format(sent_node.nodeName))
            elif node.nodeName == "#text" and node.nodeValue == '\n':
                pass
            else:
                print('Warning: element {} not recognized!'.format(node.nodeName))
    return output_sents, n_insts


class Token:
    def __init__(self, word, tag, inst_id,  lemma, sense):
        self.inst_id = inst_id
        self.word = word
        self.tag = tag
        self.lemma = lemma
        self.sense = sense
        
    def to_dict(self):
        result = {'word': self.word}
        if self.tag is not None:
            result['tag'] = self.tag
        if self.lemma is not None and self.sense is not None:
            result['sense'] = self.sense
        if self.inst_id is not None:
            result['id'] = self.inst_id
        return result


def harvest_data(xml_file, gold_file, inventory):
    xmldoc = minidom.parse(xml_file)    
    sense_map = parse_raganato_gold(gold_file, inventory)
    output_sents, n_insts = parse_raganato_xml(xmldoc, sense_map)
    return output_sents, n_insts


def harvest_multi(xml_files, gold_files):
    inventory = create_sense_inventory(gold_files)
    corpora = dict()
    for xml_file, gold_file in zip(xml_files, gold_files):
        output_sents, n_insts = harvest_data(xml_file, gold_file, inventory)
        corpora[xml_file] = {'sents': output_sents, 'n_insts': n_insts}
    result = {'inventory': inventory, 'corpora': corpora}
    return result

 
def run(xml_files, gold_files, json_output):
    """
    e.g. run(['Training_Corpora/SemCor/semcor.data.xml'],
             ['Training_Corpora/SemCor/semcor.gold.key.txt'],
             'semcor.json')

    """
    result = harvest_multi(xml_files, gold_files)
    with open(json_output, 'w') as writer:
        writer.write(json.dumps(result, indent=4))
    
    
if __name__ == '__main__':    
    root_data_dir = sys.argv[1]
    train_dir = join(root_data_dir, 'Training_Corpora')
    eval_dir = join(root_data_dir, 'Evaluation_Datasets')
    xml_files = [join(train_dir, 'SemCor/semcor.data.xml'),
                 join(eval_dir, 'semeval2007/semeval2007.data.xml')]
    gold_files = [join(train_dir, 'SemCor/semcor.gold.key.txt'),
                  join(eval_dir, 'semeval2007/semeval2007.gold.key.txt')]
    run(xml_files, gold_files, sys.argv[2])
    
