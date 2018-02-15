# -*- coding: utf-8 -*-

from xmlloader import XmlLoader
from preprocessor import Preprocessor
import numpy as np
from sklearn.linear_model import LogisticRegression

xml_loader = XmlLoader()
xml_loader.load('../../Downloads/paraphraser/paraphrases.xml')
result = xml_loader.parse()

news_corp_preprocessor = Preprocessor(result)

feature_train = list()
feature_test = list()
target_train = list()
target_test = list()

for i in range (0, 100):
    res = next(result)
    train_item = 2*[0]
    target_item = 1*[0]
    train_item[0] = news_corp_preprocessor.get_embeddings(res.string_1)
    train_item[1] = news_corp_preprocessor.get_embeddings(res.string_2)
    target_item[0] = res.value
    feature_train.append(train_item)
    target_train.append(target_item)

clf = LogisticRegression(fit_intercept=True, n_jobs=1)
clf.fit(X=feature_train, y=target_train)
print(clf.score(X=feature_test, y=target_test))
clf.predict(feature_test)