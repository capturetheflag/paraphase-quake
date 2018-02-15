# -*- coding: utf-8 -*-

#import numpy as np
#from gensim.models.fasttext import FastText
from xmlloader import XmlLoader

#test1 = FastText.load('../../Downloads/araneum_none_fasttextskipgram_300_5_2018/araneum_none_fasttextskipgram_300_5_2018.model', mmap = 'r')
#print(test1['землетрясение'])

xml_loader = XmlLoader()
xml_loader.load('../../Downloads/paraphraser/paraphrases.xml')
result = xml_loader.parse()

for res in result:
    pass