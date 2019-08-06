class SenseInstance:
    """
    A SenseInstance corresponds to a single annotation of a word sense
    in a sentence/document. It has the following fields:
        - 'tokens' (a list of strings): the BERT tokens in the sentence
        - 'pos' (integer): the position of the sense-annotated token
        - 'sense' (string): the sense of the annotated token
    
    """   
    def __init__(self, tokens, pos, sense):
        self.tokens = tokens
        self.pos = pos
        self.sense = sense