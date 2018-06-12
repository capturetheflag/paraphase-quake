# -*- coding: utf-8 -*-

from xmlloader import XmlLoader
from feature_extractor import FeatureExtractor
from preprocessor import Preprocessor
from model import Model
import numpy as np
import scipy as scipy
from sklearn.linear_model import LogisticRegression
from keras.preprocessing import sequence

LENGTH = 22
EMDEDDING_SIZE = 300

corpus_length = 6000

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
para_count = 0
non_para_count = 0
true_para = list()
false_para = list()

for i in range (0, corpus_length):
    paraphrase = next(result)
    if (paraphrase.value > 0):
        true_para.append(paraphrase)
        para_count += 1
    else:
        false_para.append(paraphrase)
        non_para_count += 1

print((para_count, non_para_count))

# 1402, 4598

true_para = true_para + true_para # over-sampling

train_data = list()
test_data = list()

dataset_oversampling = list()

for i in range (0, len(true_para)):
    dataset_oversampling.append(false_para[i])

dataset_oversampling = dataset_oversampling + true_para

np.random.shuffle(dataset_oversampling)

for i in range (0, int(len(dataset_oversampling) / 2)):
    train_data.append(dataset_oversampling[i])

for i in range (int(len(dataset_oversampling) / 2), len(dataset_oversampling)):
    test_data.append(dataset_oversampling[i])

np.random.shuffle(train_data)
np.random.shuffle(test_data)

news_corp_preprocessor = Preprocessor(result)
feature_extractor = FeatureExtractor(news_corp_preprocessor)

### Logistic Regression technique (based on cosine similarity of averaged sentence vector)

x_train = list()
y_train = list()
x_test = list()
y_test = list()

dataset_length = int(len(dataset_oversampling) / 2)

if (1 > 0):
    for i in range (0, len(train_data)):
        item = train_data[i]

        if (item.value < 0):
            print('target class expected to be 0 or 1, but got: ' + str(item.value))

        x_item = 1*[0]
        y_item = 1*[0]
        x_item[0] = feature_extractor.get_distance(item.string_1, item.string_2)
        y_item[0] = item.value

        x_train.append(x_item)
        y_train.append(y_item)

    for i in range (0, len(test_data)):
        item = test_data[i]
        x_item = 1*[0]
        y_item = 1*[0]
        x_item[0] = feature_extractor.get_distance(item.string_1, item.string_2)
        y_item[0] = item.value

        x_test.append(x_item)
        y_test.append(y_item)

    feature_train = np.array(x_train)
    target_train = np.array(y_train)
    feature_test = np.array(x_test)
    target_test = np.array(y_test)

    clf = LogisticRegression(fit_intercept=True, n_jobs=1)
    clf.fit(X=feature_train, y=target_train)
    score = clf.score(X=feature_test, y=target_test)
    print("Cosine distance (baseline). Accuracy: %.2f%%" % (score * 100))
    # clf.predict(feature_test)

### Deep neural network technique

x_train = list()
y_train = list()
x_test = list()
y_test = list()

for i in range (0, len(train_data)):
    data_item = train_data[i]
    x_item, y_item = get_feature(data_item)
    x_train.append(x_item)
    y_train.append(y_item)

for i in range (0, len(test_data)):
    data_item = test_data[i]
    x_item, y_item = get_feature(data_item)
    x_test.append(x_item)
    y_test.append(y_item)

nn_model = Model(sequence_length=LENGTH, vector_length=EMDEDDING_SIZE)
nn_model.fit(np.array(x_train), np.array(y_train), batch_size=10, epochs=10)
nn_model.predict(np.array(x_test), y_test=np.array(y_test))

# выкинуть предлоги ?? 
# стемминг