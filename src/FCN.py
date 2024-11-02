from collections import defaultdict, namedtuple
from typing import Tuple, List, Dict, Optional
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

from src.represent_tea_sample import CompoundDataSample, DataEncodeMode
from src.init_params import CYCLES, TARGET, N_CLASSES, N_BINS, X_MIN, X_MAX
from src.metrics import calc_metrics

class FCN_Model(tf.keras.Model):
    def __init__(
        self, 
        mode: DataEncodeMode, 
        n_filters_sequence: Tuple[int] = (8, 16, 32, 64),
        kernel_size_sequence: Tuple[int] = (16, 8, 4, 2),
        pool_window_size_sequence: Tuple[int] = (4, 2, 2, 2),
        input_sequence_length = N_BINS,
    ):
        super().__init__()
        self.mode = mode
        if mode == DataEncodeMode.FIRST_CYCLE_ONLY or mode == DataEncodeMode.SECOND_CYCLE_ONLY:
            n_channels_in = 8
        elif mode == DataEncodeMode.ALL_CYCLES:
            n_channels_in = 3 * 8
        else:
            assert mode == DataEncodeMode.MS
            n_channels_in = 4
        
        extractor = tf.keras.Sequential()
        extractor.add(tf.keras.Input(shape=(input_sequence_length, n_channels_in)))
        for n_filters, kernel_size, pool_window_size in zip(n_filters_sequence, 
                                                            kernel_size_sequence, 
                                                            pool_window_size_sequence):
            extractor.add(tf.keras.layers.Conv1D(n_filters, 
                                                 padding="same",
                                                 kernel_size=kernel_size, 
                                                 activation='relu'))
            extractor.add(tf.keras.layers.BatchNormalization())
            extractor.add(tf.keras.layers.MaxPool1D(pool_window_size))
        
        extractor.add(tf.keras.layers.Flatten(data_format="channels_last"))
        extractor.add(tf.keras.layers.Dropout(0.5))
        self.extractor = extractor
        
        self.final = tf.keras.layers.Dense(N_CLASSES)
        
    def call(self, input):
        cycle_input = input[CYCLES]
        inner = self.extractor(cycle_input)
        out = self.final(inner)
        return out
    
    def predict_probas(self, sample: CompoundDataSample) -> Tuple:
        nn_input = sample.to_nn_sample(X_MIN, X_MAX, N_BINS, mode=self.mode)
        input, y = preprocess(nn_input, None)
        input[CYCLES] = tf.expand_dims(tf.convert_to_tensor(input[CYCLES]), axis=0)
        out = self.call(input)
        return out[0].numpy()
    
    def predict_inner_outputs(self, sample: CompoundDataSample) -> List[tf.Tensor]:
        nn_input = sample.to_nn_sample(X_MIN, X_MAX, N_BINS, mode=self.mode)
        input, y = preprocess(nn_input, None)
        input[CYCLES] = tf.expand_dims(tf.convert_to_tensor(input[CYCLES]), axis=0)
        outputs = []
        x = input[CYCLES]
        for l in self.extractor.layers:
            x = l(x)
            if isinstance(l, tf.keras.layers.Conv1D):
                outputs.append(x)
        return outputs

def plot_history(history):
    plt.figure(figsize=(12, 8))
    plt.grid()
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="validation loss")
    plt.plot(history.history['categorical_accuracy'], label="categorical accuracy")
    plt.plot(history.history['val_categorical_accuracy'], label="validation categorical accuracy")
    plt.xlabel('epoch')

    plt.legend(loc='upper left')
    plt.show()

def train(
    checkpoint_path: Path, 
    model: FCN_Model, 
    train_dataset: tf.data.Dataset, 
    val_dataset: tf.data.Dataset,
    n_epochs: int,
):
    model.compile(optimizer='rmsprop',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    check_point = tf.keras.callbacks.ModelCheckpoint(checkpoint_path.as_posix(), 
                                                     monitor='val_categorical_accuracy',
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=0)
    return model.fit(train_dataset, 
                     epochs=n_epochs, 
                     validation_data=val_dataset,
                     callbacks=[check_point],
                     verbose=1)


def evaluate(model, val_dataset):
    model.trainable = False
    pred = []
    target = []
    for batch in val_dataset:
        pred.extend(model(batch[0]).numpy().argmax(axis=1))
        target.extend(batch[1].numpy().argmax(axis=1))
        
    calc_metrics(pred, target)
