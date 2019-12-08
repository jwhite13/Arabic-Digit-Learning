import numpy as np
import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes

        self.batch_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.conv1 = tf.keras.layers.Conv2D(16, (5, 5), padding="SAME", data_format='channels_last')
        self.conv2 = tf.keras.layers.Conv2D(20, (3, 3), padding="SAME", data_format='channels_last')
        # self.conv3 = tf.keras.layers.Conv2D(20, (3, 3), padding="SAME", data_format='channels_last')

        self.normalize1 = tf.keras.layers.BatchNormalization()
        self.normalize2 = tf.keras.layers.BatchNormalization()

        self.relu1 = tf.keras.layers.ReLU()
        self.relu2 = tf.keras.layers.ReLU()
        # self.relu3 = tf.keras.layers.ReLU()

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.dense2 = tf.keras.layers.Dense(30, activation='relu')
        self.densef = tf.keras.layers.Dense(num_classes)

        self.flatten = tf.keras.layers.Flatten()


    #@tf.function
    def call(self, data):
        data = tf.expand_dims(out[0:idx], 3)
        conv1 = self.conv1(data)
        norm1 = self.normalize1(conv1)
        relu1 = self.relu1(norm1)
        pool1 = self.pool1(relu1)

        conv2 = self.conv2(pool1)
        norm2 = self.normalize2(conv2)
        relu2 = self.relu2(norm2)
        pool2 = self.pool2(relu2)

        dense_input = self.flatten(pool2)

        dense1 = self.dense1(dense_input)
        dense2 = self.dense2(dense1)
        logits = self.densef(dense2)

        return logits

    def accuracy_function(self, prbs, labels):
        print("test")
        guesses = tf.argmax(prbs, axis=1)
        print(guesses)
        labels = tf.where(tf.equal(labels, 1))
        labels = labels[:,1]
        print(labels)
        accuracy = tf.cast(tf.equal(guesses, labels), dtype=tf.float32)
        print(accuracy)

    	# correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    	# return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        ttl_correct = tf.reduce_sum(accuracy)
        print(ttl_correct)
        return ttl_correct


    def loss(self, prbs, labels):
        # print(labels.shape)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, prbs)
        return loss
