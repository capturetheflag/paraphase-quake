# -*- coding: utf-8 -*-

from gensim.utils import simple_preprocess
from pymystem3 import Mystem
from nltk.stem import SnowballStemmer
from gensim.models.fasttext import FastText
import numpy as np
import nltk
from nltk.corpus import stopwords
import urllib.request
import os
import tarfile

EMBEDDINGS_URL = 'http://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextskipgram_300_5_2018.tgz'
EMBEDDINGS_FOLDER = './model/'
EMBEDDINGS_FILE = 'araneum_none_fasttextskipgram_300_5_2018.model'

nltk.download('stopwords')
nltk.download('punkt')

if (not os.path.exists(EMBEDDINGS_FOLDER + EMBEDDINGS_FILE)):
    urllib.request.urlretrieve(EMBEDDINGS_URL, EMBEDDINGS_FOLDER)
    tar = tarfile.open(EMBEDDINGS_FILE, "r:gz")
    tar.extractall(EMBEDDINGS_FOLDER)
    tar.close()

class Preprocessor:
    def __init__(self, paraphrases):
        self.embeddings = FastText.load(EMBEDDINGS_FOLDER + EMBEDDINGS_FILE, mmap = 'r')
        self.paraphrases = paraphrases
        #self.stemmer = SnowballStemmer('russian')
        self.stemmer = Mystem()
        
    def get_sentence_embedding(self, sentence):
        words = self.tokenize(sentence)
        embeddings = list()
        for word in words:
            if (word in self.embeddings):
                embeddings.append(self.embeddings[word])
        return np.sum(embeddings, axis=0)

    def get_words_embeddings(self, sentence):
        words = self.tokenize(sentence)
        embeddings = list()
        for word in words:
            if (word in self.embeddings):
                embeddings.append(self.embeddings[word])
        return embeddings

    def tokenize(self, sentence):
        tokenize = lambda x: nltk.word_tokenize(x)
        words = tokenize(sentence)

        stop_words = stopwords.words('russian')
        stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на', '.', ',', '\'', '"', ":"])
        words = [i for i in words if ( i not in stop_words )]
        
        normalized_words = list()

        for word in words:
            normalized_words.append(self.stemmer.lemmatize(word)[0])

        return normalized_words
        

# Pymorphic tool for lemmas and pos-taging 
# IAM's tool (Lyubov Yurievna can share)