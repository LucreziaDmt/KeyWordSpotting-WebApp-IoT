import tensorflow as tf
from scipy import signal
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import time
from signalgenerator import SignalGenerator
from models import MLP, ConvNet, DS_CNN

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
zip_path = tf.keras.utils.get_file(
     origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
     fname='mini_speech_commands.zip',
     extract=True,
     cache_dir='.',
     cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')
filenames = tf.io.gfile.glob(str(data_dir) + "/*/*")
filenames = tf.random.shuffle(filenames)

n = len(filenames)

train_data = filenames[:int(n * 0.8)]
val_data = filenames[int(n * 0.8):int(n * 0.9)]
test_data = filenames[int(n * 0.9):]

LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
LABELS = LABELS[LABELS != 'README.md']
n_classes = len(LABELS)

#['right', 'up', 'left', 'stop', 'no', 'go', 'yes', 'down'])

frame_length = int(16e3*40e-3)
frame_step = int(16e3*20e-3)
signal_generator = SignalGenerator(LABELS, 16000, frame_length, frame_step, num_mel_bins=40,
                                   lower_frequency=20, upper_frequency=4000, num_coefficients=10,
                                   mfcc=True)
strides = [2,1]

train_ds = signal_generator.make_dataset(train_data, True)
test_ds = signal_generator.make_dataset(test_data, False)
val_ds = signal_generator.make_dataset(val_data, False)

#Train the MultiLayer Perceptron
mlp = MLP()
mlp.train(train_ds, val_ds, 20)
filename = 'Model_mlp'

mlp._model().save(filename)

#Train the Convolutional NN
cnn = ConvNet(strides)
cnn.train(train_ds, val_ds, 20)
filename = 'Model_cnn'
cnn._model().save(filename)

#Train the DS Convolutional NN
dscnn = DS_CNN(strides)
dscnn.train(train_ds, val_ds, 20)
filename = 'Model_dscnn'
dscnn._model().save(filename)

