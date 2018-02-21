# -*- coding: utf-8 -*-

from gensim.utils import simple_preprocess
from gensim.models.fasttext import FastText
import numpy as np

class Preprocessor:
    def __init__(self, paraphrases):
        self.embeddings = FastText.load(
            '../../Downloads/araneum_none_fasttextskipgram_300_5_2018/araneum_none_fasttextskipgram_300_5_2018.model', 
            mmap = 'r')
        self.paraphrases = paraphrases
        
    def get_embeddings(self, sentence):
        words = self.tokenize(sentence)
        embeddings = list()
        for word in words:
            embeddings.append(self.embeddings[word])
        return np.sum(embeddings, axis=0)
        
    def tokenize(self, sentence):
        tokenize = lambda x: simple_preprocess(x)
        return tokenize(sentence)
        
        