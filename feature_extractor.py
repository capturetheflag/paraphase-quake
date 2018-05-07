# -*- coding: utf-8 -*-

from scipy.spatial import distance

class FeatureExtractor:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def get_distance(self, sentence1, sentence2):
        sentenceEmbd1 = self.preprocessor.get_sentence_embedding(sentence1)
        sentenceEmbd2 = self.preprocessor.get_sentence_embedding(sentence2)
        return distance.cosine(sentenceEmbd1, sentenceEmbd2)