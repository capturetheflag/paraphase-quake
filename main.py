# -*- coding: utf-8 -*-

from xmlloader import XmlLoader
from feature_extractor import FeatureExtractor
from preprocessor import Preprocessor
from model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression

xml_loader = XmlLoader()
xml_loader.load('../../Downloads/paraphraser/paraphrases.xml')
result = xml_loader.parse()

news_corp_preprocessor = Preprocessor(result)
feature_extractor = FeatureExtractor(news_corp_preprocessor)

x_train = list()
y_train = list()
x_test = list()
y_test = list()

### Logistic Regression technique

for i in range (0, 3000):
    res = next(result)

    if (res.value < 0):
        print('target class expected to be 0 or 1, but got: ' + str(res.value))

    x_item = 1*[0]
    y_item = 1*[0]
    x_item[0] = feature_extractor.get_distance(res.string_1, res.string_2)
    y_item[0] = res.value
    x_train.append(x_item)
    y_train.append(y_item)

for i in range (4000, 5000):
    res = next(result)
    x_item = 1*[0]
    y_item = 1*[0]
    x_item[0] = feature_extractor.get_distance(res.string_1, res.string_2)
    y_item[0] = res.value
    x_test.append(x_item)
    y_test.append(y_item)

feature_train = np.array(x_train)
target_train = np.array(y_train)
feature_test = np.array(x_test)
target_test = np.array(y_test)

clf = LogisticRegression(fit_intercept=True, n_jobs=1)
clf.fit(X=feature_train, y=target_train)
print(clf.score(X=feature_test, y=target_test))
clf.predict(feature_test)

### Deep neural network technique

#nn_model = Model()
#nn_model.fit(feature_train, target_train, 1)
#nn_model.predict(feature_test)