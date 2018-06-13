# -*- coding: utf-8 -*-

from xmlloader import XmlLoader
from feature_extractor import FeatureExtractor
from preprocessor import Preprocessor
from model import Model
import numpy as np
import scipy as scipy
from sklearn.linear_model import LogisticRegression
from keras.preprocessing import sequence

SEQUENCE_LENGTH = 10
EMDEDDING_SIZE = 300
CORPUS_LENGTH = 6000

def get_feature (item):
    if (data_item.value < 0):
        print('target class expected to be 0 or 1, but got: ' + str(item.value))

    words1 = news_corp_preprocessor.get_words_embeddings(item.string_1)
    words2 = news_corp_preprocessor.get_words_embeddings(item.string_2)
    x_item = np.empty([SEQUENCE_LENGTH, SEQUENCE_LENGTH])

    a = np.zeros((SEQUENCE_LENGTH, EMDEDDING_SIZE))
    b = np.zeros((SEQUENCE_LENGTH, EMDEDDING_SIZE))

    for i in range (0, SEQUENCE_LENGTH):
        if (i < len(words1)):
            a[i] = words1[i]
        if (i < len(words2)):
            b[i] = words2[i]

    for i in range (0, SEQUENCE_LENGTH):
        for j in range (0, SEQUENCE_LENGTH):
            x_item[i][j] = feature_extractor.get_word_similarity(a[i], b[j])

    x_item = np.reshape(x_item, (SEQUENCE_LENGTH, SEQUENCE_LENGTH))
    y_item = 1*[0]
    y_item[0] = data_item.value

    return (x_item, y_item)

xml_loader = XmlLoader()
xml_loader.load('./corpus/paraphrases.xml')
result = xml_loader.parse()

#### Count number of paraphrases and non-paraphrases in the dataset
para_count = 0
non_para_count = 0
true_para = list()
false_para = list()

for i in range (0, CORPUS_LENGTH):
    paraphrase = next(result)
    if (paraphrase.value > 0):
        true_para.append(paraphrase)
        para_count += 1
    else:
        false_para.append(paraphrase)
        non_para_count += 1

print((para_count, non_para_count))

# 1402, 4598

train_data = list()
test_data = list()
dataset = list()

for i in range (0, len(true_para)):
    dataset.append(true_para[i])
    dataset.append(false_para[i])

np.random.shuffle(dataset)

for i in range (0, int(len(dataset) * 0.8)):
    train_data.append(dataset[i])

for i in range (int(len(dataset) * 0.8), len(dataset)):
    test_data.append(dataset[i])

# Over sampling

for i in range (0, len(true_para)):
    found = False
    sample = true_para[i]
    for j in range (0, len(test_data)):
        if (sample.id == test_data[j].id):
            found = True
            break
    if (not found):
        train_data.append(sample)
        train_data.append(false_para[i + len(true_para)])

np.random.shuffle(train_data)
np.random.shuffle(test_data)

news_corp_preprocessor = Preprocessor(result)
feature_extractor = FeatureExtractor(news_corp_preprocessor)

### Logistic Regression technique (based on cosine similarity of averaged sentence vector)

x_train = list()
y_train = list()
x_test = list()
y_test = list()

if (1 > 0):
    for i in range (0, len(train_data)):
        item = train_data[i]

        if (item.value < 0):
            print('target class expected to be 0 or 1, but got: ' + str(item.value))

        x_item = 1*[0]
        y_item = 1*[0]
        x_item[0] = feature_extractor.get_distance(item.string_1, item.string_2)
        y_item = item.value

        x_train.append(x_item)
        y_train.append(y_item)

    for i in range (0, len(test_data)):
        item = test_data[i]
        x_item = 1*[0]
        y_item = 1*[0]
        x_item[0] = feature_extractor.get_distance(item.string_1, item.string_2)
        y_item = item.value

        x_test.append(x_item)
        y_test.append(y_item)

    clf = LogisticRegression(fit_intercept=True, n_jobs=1)
    clf.fit(X=np.array(x_train), y=np.array(y_train))
    score = clf.score(X=np.array(x_test), y=np.array(y_test))
    print("Cosine distance (baseline). Accuracy: %.2f%%" % (score * 100))
    # clf.predict(np.array(y_test))

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

nn_model = Model(sequence_length=SEQUENCE_LENGTH, vector_length=SEQUENCE_LENGTH)
nn_model.fit(np.array(x_train), np.array(y_train), batch_size=10, epochs=20)
nn_model.predict(np.array(x_test), y_test=np.array(y_test))