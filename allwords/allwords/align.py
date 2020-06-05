    
def align(orig_toks, new_toks, special_toks = ['[CLS]', '[SEP]']):
    def standardize(token):
        token = token.lower()
        if token.startswith('##'):
            return standardize(token[2:])
        else:
            return token.replace('_', ' ')
    
    def match(tok, tokstream):
        i = 0
        while i < len(tok) and i < len(tokstream) and tok[i] == tokstream[i]:
            i += 1
        return i
    
    def increment_new_tok_index(new_toks, new_tok_index, special_toks):
        new_tok_index += 1
        while (new_tok_index < len(new_toks) 
               and new_toks[new_tok_index] in special_toks):
            new_tok_index += 1
        return new_tok_index

    special_toks = [standardize(tok) for tok in special_toks]
    orig_toks = [standardize(tok) for tok in orig_toks]
    new_toks = [standardize(tok) for tok in new_toks]
    new_tok_index = increment_new_tok_index(new_toks, -1, special_toks)
    alignment = []
    for orig_tok in orig_toks:
        span_start = new_tok_index
        new_tok = new_toks[new_tok_index]
        while len(orig_tok) > 0:
            num_matched = match(orig_tok, new_toks[new_tok_index])
            orig_tok = orig_tok[num_matched:]
            new_tok = new_tok[num_matched:]
            if len(new_tok) > 0 and len(orig_tok) > 0:
                #print('failed to match: "{}" and "{}"'.format(orig_tok, 
                #                                              new_tok))
                return None
            if len(new_tok) == 0 and len(orig_tok) > 0:
                new_tok_index = increment_new_tok_index(new_toks, 
                                                        new_tok_index, 
                                                        special_toks)
                if new_tok_index < len(new_toks):
                    new_tok = new_toks[new_tok_index]
                if len(orig_tok) > 0 and orig_tok[0] == ' ':
                    orig_tok = orig_tok[1:]
        span_end = new_tok_index + 1
        new_tok_index = increment_new_tok_index(new_toks, 
                                                new_tok_index, 
                                                special_toks)        
        #print('{}: {}'.format(tok_copy, new_toks[span_start:span_end]))
        alignment.append((span_start, span_end))
    new_tok_index = increment_new_tok_index(new_toks, new_tok_index, 
                                            special_toks)
    assert(new_tok_index == len(new_toks) + 1)
    return alignment
    