import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Dense,
    LSTM,
    Input,
    Bidirectional,)
from tensorflow.keras.metrics import BinaryCrossentropy as bce_metric, BinaryAccuracy, Precision, Recall, AUC, F1Score

def load_voice_softmax(n_features):
    ### softmax regression
    model = Sequential(name='voice-softmax')

    model.add(Input(shape=(n_features, ), dtype='int64', name='input-layer'))
    model.add(Dense(units=n_features))
    model.add(Activation(activation=tf.nn.softmax))



def load_voice_lstm(window_size=1000, n_a=128, dense_drop_prob=0.2, rnn_drop_prob=0.2, n_units=1, **kwargs):
    ### LSTM layers ###
    model = Sequential(name='voice-lstm')

    # shape will m x 4 since 4 is the maximum length of a name
    model.add(Input(shape=(window_size, 1), dtype='int64', name='input-layer'))

    # initialize lstm layers
    model.add(Bidirectional(LSTM(units=n_a, recurrent_dropout=rnn_drop_prob, return_sequences=False)))
    model.add(Dropout(rate=dense_drop_prob))
    model.add(Dense(units=n_units))
    model.add(Activation(activation=tf.nn.softmax))

    return model
