import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Dense,
    LSTM,
    Input,
    Bidirectional,)

def load_voice_softmax(n_features: int, n_units: int, **kwargs):
    print(f"kwargs: n_features = {n_features}, n_units = {n_units}")

    ### softmax regression
    model = Sequential(name='voice-softmax')

    model.add(Input(shape=(n_features, ), dtype='int64', name='input-layer'))
    model.add(Dense(units=n_units))
    model.add(Activation(activation=tf.nn.softmax))

    return model



def load_voice_lstm(window_size: int, n_a: int=128, dense_drop_prob: float=0.2, rnn_drop_prob: float=0.2, n_units: int=3, **kwargs):
    print(f"kwargs: window_size = {window_size}, n_a = {n_a}, dense_drop_prob = {dense_drop_prob}, rnn_drop_prob = {rnn_drop_prob}, n_units = {n_units}")

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
