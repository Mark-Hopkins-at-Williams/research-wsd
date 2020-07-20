from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
import random
from reed_wsd.allwords.bert import tokenize_with_target


def wn_definition_with_target(tokenizer, wn_lemma):
    pos = wn_lemma.synset().pos()
    defn = wn_lemma.synset().definition()
    if pos == 'n':
        gloss = wn_lemma.name() + ' is ' + defn
        rep_span = [1, 1 + len(tokenizer.tokenize(wn_lemma.name()))]
    if pos in ['a', 's', 'r']:
        gloss = wn_lemma.name() + ' means ' + defn
        rep_span = [1, 1 + len(tokenizer.tokenize(wn_lemma.name()))]
    if pos == 'v':
        gloss = 'to ' + wn_lemma.name() + ' is to ' + defn
        rep_span = [2, 2 + len(tokenizer.tokenize(wn_lemma.name()))]
    return gloss, rep_span



def wn_example(lemma, wordform, tokenizer, rand=False):
    """
    Takes the given nltk.wordnet lemma (e.g. 'be%2:42:03::') and a string
    form of that lemma (e.g. "is") and finds an example sentence from
    wordnet.

    """
    lemmatizer = WordNetLemmatizer()

    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith('J'):
            return wn.ADJ
        elif nltk_tag.startswith('V'):
            return wn.VERB
        elif nltk_tag.startswith('N'):
            return wn.NOUN
        elif nltk_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    def lemmatize_sentence(sentence):
        """
        takes a str sentence and return a list of lemmatized tokens
        """
        #tokenize the sentence and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
        #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                #if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                #else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return lemmatized_sentence
    syns_egs = lemma.synset().examples()
    syns_egs_tks = [word_tokenize(eg) for eg in syns_egs]
    lemmatized_syns_egs = [lemmatize_sentence(sent) for sent in syns_egs]
    tgt_lemma = lemmatize_sentence(wordform)[0]
    example_pairs = []
    for i, eg in enumerate(lemmatized_syns_egs):
        if tgt_lemma in eg:
            index = eg.index(tgt_lemma)
            example_pairs.append([i, index])
    if len(example_pairs) == 0:
        gloss, span = wn_definition_with_target(tokenizer, lemma)
        return gloss, span
    if rand:
        random_pair = random.sample(example_pairs, 1)[0]
    else:
        random_pair = example_pairs[0]
    _, span = tokenize_with_target(tokenizer, syns_egs_tks[random_pair[0]], random_pair[1])
    return syns_egs[random_pair[0]], span

def random_wn_example(lemma, word, tokenizer):
    return wn_example(lemma, word, tokenizer, rand=True)
