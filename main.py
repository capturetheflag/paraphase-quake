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

# Logistic Regression technique

for i in range (0, 3000):
    res = next(result)
    x_item = 1*[0]
    y_item = 1*[0]
    x_item[0] = feature_extractor.get_distance(res.string_1, res.string_2)
    
    if (res.value < 0):
        print('target class expected to be 0 or 1, but got: ' + res.value)
    
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

# Deep neural network technique

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

#nn_model = Model()
#nn_model.fit(feature_train, target_train, 1)
#nn_model.predict(feature_test)
