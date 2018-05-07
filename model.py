# -*- coding: utf-8 -*-

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Embedding, Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
import numpy as np

class Model:
    EMBEDDING_SIZE = 300
    MAX_LENGTH = 20

    def __init__(self):
        self.model = Sequential()
        self.model.add(Embedding(2, self.EMBEDDING_SIZE, input_length=self.MAX_LENGTH))
        self.model.add(Conv1D(64, 3, padding='same'))
        self.model.add(Conv1D(32, 3, padding='same'))
        self.model.add(Conv1D(16, 3, padding='same'))
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(180,activation='sigmoid'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1,activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x_train, y_train, batch_size, epochs=3):
        x_train = sequence.pad_sequences(x_train, maxlen=self.MAX_LENGTH)
        tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
        
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, callbacks=[tensorBoardCallback])

    def predict(self, x_test):
        return self.model.predict(x_test)



### TODO
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