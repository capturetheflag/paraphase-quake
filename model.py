from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np

class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv1D(32, 5, strides=1, activation='relu', input_shape=(2, 300)))
        self.model.add(MaxPooling1D(pool_size=2, strides=2))
        self.model.add(Conv1D(64, 3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        #self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x_train, y_train, batch_size, epochs = 3):
        x_train = np.expand_dims(x_train, axis=2)

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)

    def predict(self, x_test):
        return self.model.predict(x_test)