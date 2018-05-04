# -*- coding: utf-8 -*-

from xmlloader import XmlLoader
from preprocessor import Preprocessor
from model import Model
import numpy as np
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression

## Keras setup
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

xml_loader = XmlLoader()
xml_loader.load('../../Downloads/paraphraser/paraphrases.xml')
result = xml_loader.parse()

news_corp_preprocessor = Preprocessor(result)

feature_train = list()
feature_test = list()
target_train = list()
target_test = list()

for i in range (0, 3000):
    res = next(result)
    train_item = 2*[0]
    target_item = 1*[0]
    sentenceEmbd1 = news_corp_preprocessor.get_embeddings(res.string_1)
    sentenceEmbd2 = news_corp_preprocessor.get_embeddings(res.string_2)
    train_item[0] = distance.cosine(sentenceEmbd1, sentenceEmbd2)
    target_item[0] = res.value
    feature_train.append(train_item)
    target_train.append(target_item)

for i in range (4000, 5000):
    res = next(result)
    train_item = 2*[0]
    target_item = 1*[0]
    sentenceEmbd1 = news_corp_preprocessor.get_embeddings(res.string_1)
    sentenceEmbd2 = news_corp_preprocessor.get_embeddings(res.string_2)
    train_item[0] = distance.cosine(sentenceEmbd1, sentenceEmbd2)
    target_item[0] = res.value
    feature_test.append(train_item)
    target_test.append(target_item)

feature_train = np.array(feature_train)
target_train = np.array(target_train)

feature_test = np.array(feature_test)
target_test = np.array(target_test)

# clf = LogisticRegression(fit_intercept=True, n_jobs=1)
# clf.fit(X=feature_train, y=target_train)
# print(clf.score(X=feature_test, y=target_test))
# clf.predict(feature_test)

# препроцессинг (убрать ненужные слова, лемматизировать)
# tf-idf
# описать корпус: количество слов в корпусе, размер
# 
# на выходе должно показываться n-лучших перифраз по заданной фразе
# описать какие типы перифраз бывает
# предобработанный текст - сохранять на диск
# описать каждое изменение параметров (замена слоформ на лексемы в качестве признака)

# 1. as for the baseline - use word2vec model and fastText model
# 2. compare it, usi g logistic regresssion
# 3. train own word embeddings on test data and see how it goes
# 4. compare those and make a conclusion
# 5. choose one among them and use in NN model (like CNN)
# 6. Fix CNN model (16 hours)

# LSTM for sequence classification in the IMDB dataset

# fix random seed for reproducibility
# np.random.seed(7)
# # load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
# # truncate and pad input sequences
# max_review_length = 
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# # create the model
# embedding_vector_length = 32
# model = Sequential()
# model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

nn_model = Model()
nn_model.fit(feature_train, target_train, 1)
nn_model.predict(feature_test)
