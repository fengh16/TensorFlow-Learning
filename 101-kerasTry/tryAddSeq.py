import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, units=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])