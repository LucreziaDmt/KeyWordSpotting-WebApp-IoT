import tensorflow as tf
from tensorflow import keras

class MLP:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(49, 10, 1)),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dense(units=8),
            tf.keras.layers.Softmax()])

    def _model(self):
        return self.model

    def summary(self):
        return self.model.summary()

    def train(self, Train, Val, epochs):
        self.model.compile(optimizer=tf.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        history = self.model.fit(Train, epochs=epochs, validation_data=Val)

    def test(self, Test):
        test_loss, test_accuracy = self.model.evaluate(Test)
        return test_accuracy

class ConvNet:
    def __init__(self,strides):
        self.model = keras.Sequential([
            tf.keras.layers.Conv2D(filters=128, kernel_size =[3,3],
                                   strides=strides, use_bias = False, input_shape=(49, 10, 1)),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3],
                                   strides=[1,1], use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1],
                                   strides=[1, 1], use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=8),
            tf.keras.layers.Softmax()])

    def _model(self):
        return self.model

    def summary(self):
        return self.model.summary()

    def train(self, Train, Val, epochs):
        self.model.compile(optimizer=tf.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        history = self.model.fit(Train, epochs=epochs, validation_data=Val)
    def test(self, Test):
        test_loss, test_accuracy = self.model.evaluate(Test)
        return test_accuracy


class DS_CNN:
    def __init__(self, strides):
        self.model = keras.Sequential([
            tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3],
                                   strides=strides, use_bias=False, input_shape=(49, 10, 1)),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size = [3,3], strides = [1,1],
                                            use_bias = False),
            tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1],
                                   strides=[1, 1], use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                            use_bias=False),
            tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1],
                                   strides=[1, 1], use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D(),          
            tf.keras.layers.Dense(units=8),
            tf.keras.layers.Softmax()])

    def _model(self):
        return self.model

    def summary(self):
        return self.model.summary()

    def train(self, Train, Val, epochs):
        self.model.compile(optimizer=tf.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=tf.keras.metrics.SparseCategoricalAccuracy())
        history = self.model.fit(Train, epochs=epochs, validation_data=Val)

    def test(self, Test):
        test_loss, test_accuracy = self.model.evaluate(Test)
        return test_accuracy

