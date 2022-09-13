import warnings

import tensorflow as tf

warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)


class CustomArchitecture(tf.keras.layers.Layer):
    def __init__(self,
                 input_size: int,
                 theta_size: int,
                 horizon: int,
                 n_neurons: int,
                 n_layers: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        self.LSTM1 = tf.keras.layers.LSTM(128, return_sequences=True, activation='relu')
        self.LSTM2 = tf.keras.layers.LSTM(128, return_sequences=True, activation='relu')
        self.LSTM3 = tf.keras.layers.LSTM(128, return_sequences=True, activation='relu')
        self.LSTM4 = tf.keras.layers.LSTM(128, return_sequences=False, activation='relu')

        self.intermediate_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs):
        x = inputs

        x = tf.expand_dims(x, axis=1)

        x = self.LSTM1(x)
        x = self.LSTM2(x)
        x = self.LSTM3(x)
        x = self.LSTM4(x)

        combinedForecast = self.intermediate_layer(x)

        residual_inp, global_residual_op = combinedForecast[:, :self.input_size], combinedForecast[:, -self.horizon:]
        return residual_inp, global_residual_op
