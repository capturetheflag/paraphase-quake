# -*- coding: utf-8 -*-

from gensim.utils import simple_preprocess

class Preprocessor:
    def __init__(self, text_corpus):
        self.corpus = text_corpus
        
    def get_embeddings(self):
        return []
    
    def tokenize(self, sentence):
        tokenize = lambda x: simple_preprocess(x)
        return tokenize(sentence)
        
        