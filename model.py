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
    MAX_LENGTH = 50

    def __init__(self):
        self.model = Sequential()
        self.model.add(Embedding(10, self.EMBEDDING_SIZE, input_length=self.MAX_LENGTH))
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