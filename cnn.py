import numpy as np
import tensorflow as tf

class CNN(tf.keras.Model):
	def __init__(self, num_classes):
		super(CNN, self).__init__()
		self.num_classes = num_classes

		self.batch_size = 100
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

		self.normalize = tf.keras.layers.BatchNormalization()
		self.flatten = tf.keras.layers.Flatten()
		self.conv1 = tf.keras.layers.Conv2D(16, (5, 5), padding="SAME", data_format='channels_last')
		self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
		self.conv2 = tf.keras.layers.Conv2D(20, (3, 3), padding="SAME", data_format='channels_last')
		self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
		self.conv3 = tf.keras.layers.Conv2D(20, (3, 3), padding="SAME", data_format='channels_last')
		self.dense1 = tf.keras.layers.Dense(50, activation='relu')
		self.dense2 = tf.keras.layers.Dense(30, activation='relu')
		self.densef = tf.keras.layers.Dense(num_classes)
	
	#@tf.function
	def call(self, data):
		conv1 = tf.nn.relu(self.normalize(self.conv1(data)))
		pool1 = self.pool1(conv1)
		conv2 = tf.nn.relu(self.normalize(self.conv2(pool1)))
		dense_input = self.flatten(conv2)
		dense1 = self.dense1(dense_input)
		dense2 = self.dense2(dense1)
		logits = self.densef(dense2)
		return logits

	def accuracy_function(self, prbs, labels):
		guesses = tf.argmax(prbs, axis=1)
		labels = tf.where(tf.equal(labels, 1))
		labels = labels[:,1]
		accuracy = tf.cast(tf.equal(guesses, labels), dtype=tf.float32)
		return accuracy


	def loss(self, prbs, labels):
		loss = tf.nn.softmax_cross_entropy_with_logits(labels, prbs)
		return loss

