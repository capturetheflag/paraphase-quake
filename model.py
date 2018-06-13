# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import TensorBoard
import numpy as np

class Model:
    def __init__(self, sequence_length=18, vector_length=100):
        self.model = Sequential()
        self.model.add(Conv1D(32, 3, padding='same', input_shape=(sequence_length, vector_length)))
        self.model.add(Dropout(0.2))
        self.model.add(MaxPooling1D(4))
        self.model.add(LSTM(100))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x_train, y_train, batch_size, epochs=5, verbose=1):
        tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)      
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=False, callbacks=[tensorBoardCallback])

    def predict(self, x_test, y_test):
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        print("Layered DNN. Accuracy: %.2f%%" % (scores[1]*100))



### TODO
# tf-idf для русского языка
# на выходе должно показываться n-лучших перифраз по заданной фразе
# описать какие типы перифраз бывает
# предобработанный текст - сохранять на диск ?
# описать каждое изменение параметров (замена слоформ на лексемы в качестве признака)

# 1. as for the baseline - use word2vec model and fastText model
# 2. compare it, using logistic regresssion
# 3. train own word embeddings on test data and see how it goes
# 4. compare those and make a conclusion
# 5. choose one among them and use in NN model (like CNN)
# 6. Fix CNN model (16 hours -hahaha - that was over optimistic even for me)

# provide description