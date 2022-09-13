import warnings

import tensorflow as tf

from config import *
from model import CustomArchitecture
from preprocessing import train_dataset, test_dataset

warnings.filterwarnings("ignore")

import logging

logging.disable(logging.CRITICAL)

from tensorflow.keras import layers

residual_inp = tf.eye(1, 14)

tf.random.set_seed(42)

custom_arch = CustomArchitecture(input_size=INPUT_SIZE,
                                 intermediate_layer_size=INTERMEDIATE_LAYER_SIZE,
                                 horizon=HORIZON,
                                 n_neurons=N_NEURONS,
                                 n_layers=N_LAYERS,
                                 name="InitialBlock")

stack_input = layers.Input(shape=INPUT_SIZE, name="stack_input")

residuals, forecast = custom_arch(stack_input)

residuals = layers.subtract([stack_input, residual_inp], name=f"subtract_00")

for i, _ in enumerate(range(N_STACKS - 1)):
    residual_inp, block_forecast = CustomArchitecture(
        input_size=INPUT_SIZE,
        intermediate_layer_size=INTERMEDIATE_LAYER_SIZE,
        horizon=HORIZON,
        n_neurons=N_NEURONS,
        n_layers=N_LAYERS,
        name=f"CustomArchitecture_{i}"
    )(residuals)

    residuals = layers.subtract([residuals, residual_inp], name=f"subtract_{i}")
    forecast = layers.add([forecast, block_forecast], name=f"add_{i}")

model = tf.keras.Model(inputs=stack_input,
                              outputs=forecast,
                              name="CustomArchitecture")

model.compile(loss="mae",
                     optimizer=tf.keras.optimizers.Adam(0.001),
                     metrics=["mae", "mse"])

model.fit(train_dataset,
                 epochs=N_EPOCHS,
                 validation_data=test_dataset,
                 verbose=1,

                 callbacks=[
                     tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])
