# -*- coding: utf-8 -*-

from scipy.spatial import distance
import numpy as np

class FeatureExtractor:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def get_word_similarity(self, word1, word2):
        # argument check for zero vector
        if (not word1.any() or not word2.any()):
            return 0
        
        return distance.cosine(word1, word2)

    def get_distance(self, sentence1, sentence2):
        sentenceEmbd1 = self.preprocessor.get_sentence_embedding(sentence1)
        sentenceEmbd2 = self.preprocessor.get_sentence_embedding(sentence2)
        return distance.cosine(sentenceEmbd1, sentenceEmbd2)