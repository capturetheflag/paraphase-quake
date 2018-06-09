# -*- coding: utf-8 -*-

from xmlloader import XmlLoader
from feature_extractor import FeatureExtractor
from preprocessor import Preprocessor
from model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.preprocessing import sequence

dataset_length = 3000
LENGTH = 22
EMDEDDING_SIZE = 300

def get_feature (data_item):
    if (data_item.value < 0):
        print('target class expected to be 0 or 1, but got: ' + str(data_item.value))

    x_item = news_corp_preprocessor.get_words_embeddings(data_item.string_1) + news_corp_preprocessor.get_words_embeddings(data_item.string_2)
    x_item = np.array(x_item)[:LENGTH]

    a = np.zeros((LENGTH, EMDEDDING_SIZE))

    for i in range (0, LENGTH):
        if (i < len(x_item)):
            a[i] = x_item[i]

    x_item = np.reshape(a, (LENGTH, EMDEDDING_SIZE, 1))

    y_item = 1*[0]
    y_item[0] = data_item.value

    return (x_item, y_item)

xml_loader = XmlLoader()
xml_loader.load('../../Downloads/paraphraser/paraphrases.xml')
result = xml_loader.parse()

#### Count number of paraphrases and non-paraphrases in the dataset
# para_count = 0
# non_para_count = 0
# for i in range (0, 2 * dataset_length):
#     paraphrase = next(result)
#     if (paraphrase.value > 0):
#         para_count += 1
#     else:
#         non_para_count += 1

# print((para_count, non_para_count))

news_corp_preprocessor = Preprocessor(result)
feature_extractor = FeatureExtractor(news_corp_preprocessor)

### Logistic Regression technique

x_train = list()
y_train = list()
x_test = list()
y_test = list()

if (1 < 0):
    for i in range (0, dataset_length):
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

x_train = list()
y_train = list()
x_test = list()
y_test = list()

#x_train = np.zeros((dataset_length - 1, 10, EMDEDDING_SIZE, 1))
#y_train = np.zeros((1, 4), dtype=np.bool_)

for i in range (0, dataset_length):
    data_item = next(result)
    x_item, y_item = get_feature(data_item)
    x_train.append(x_item)
    y_train.append(y_item)

for i in range (dataset_length, 2 * dataset_length):
    data_item = next(result)
    x_item, y_item = get_feature(data_item)
    x_test.append(x_item)
    y_test.append(y_item)

nn_model = Model(sequence_length=LENGTH, vector_length=EMDEDDING_SIZE)

# index = 0
# for train_sample in x_train:
#     train_result = np.zeros(10)
#     for i in range (0, MAX_LENGTH):
#         train_result[i] = y_train[index]
  
#     nn_model.fit(train_sample, train_result, batch_size=1, epochs=3)
#     index += 1

nn_model.fit(np.array(x_train), np.array(y_train), batch_size=10, epochs=10)
nn_model.predict(np.array(x_test), y_test=np.array(y_test))