import tensorflow as tf
import numpy as np
from preprocess import get_data
import time
import sys

##differences from class rnn:
    ## pass hidden state betweeen batches
    ## initial state using generate random weight situation
    ## only train readout layer (what is readout layer)
    ##NOTE: tf rnn cells only process one timestep at a time, which could be what we're looking for

##at meeting with TA:
    ## ask about differences between vanilla rnn and this one (how to implement?)
    ## ask about this vs esn file (rip)
    ## ask about logistics of deep learning day -- who judges, what's the deal
    ## ask about poster qualitative vs quantitative result ideas

class Model(tf.keras.Model):
    def __init__(self):

        """
        The Model class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(.01)

        self.num_classes = 10
        self.batch_size = 128
        self.rnn_size = 512
        self.dense_1_sz = 64

        #embedding??
        self.gru = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True, trainable=False, recurrent_dropout = 0.9)
        #add more grus?
        self.dense_1 = tf.keras.layers.Dense(self.dense_1_sz)
        self.dense_f = tf.keras.layers.Dense(self.num_classes)
        self.softmax = tf.keras.layers.Softmax()

    def makeResevoir(self, inputs, initial_state):
        rnn_layer = self.gru(inputs)
        rnn_result = rnn_layer[0] #100x93x256
        final_state = rnn_layer[-1] #100x256
        return final_state

    def call(self, resevoir):
        # inputs = tf.reshape(inputs, [100, 93, -1])
        probs = self.dense_1(resevoir)
        probs = self.dense_f(probs)
        return probs

    def loss(self, labels, probs):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, probs)
        return loss


    def accuracy_function(self, prbs, labels):
        guesses = tf.argmax(prbs, axis=1)
        labels = tf.where(tf.equal(labels, 1))
        labels = labels[:,1]
        accuracy = tf.cast(tf.equal(guesses, labels), dtype=tf.float32)

        return tf.reduce_sum(accuracy)


def train(model, train_data, train_labels):
    indices = list(range(len(train_labels)))
    indices = tf.random.shuffle(indices)
    tf.gather(train_data, indices)
    tf.gather(train_labels, indices)

    leng = int(len(train_labels) * 1)
    ttl_loss = 0

    for idx in range(0, leng, model.batch_size):
        data_batch = train_data[idx:idx + model.batch_size]
        label_batch = train_labels[idx:idx + model.batch_size]    #
        if len(label_batch) == model.batch_size:
            resevoir = model.makeResevoir(data_batch, None)
            with tf.GradientTape() as tape:
                logits = model.call(resevoir) ##this will be different, check this out
                loss = model.loss(label_batch, logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            ttl_loss += tf.reduce_sum(loss)

    return ttl_loss/leng


def test(model, test_data, test_labels):
    correct = 0
    total = 0

    for idx in range(0, len(test_labels), model.batch_size):
        data_batch = test_data[idx:idx + model.batch_size]
        label_batch = test_labels[idx:idx + model.batch_size]

        resevoir = model.makeResevoir(data_batch, None)
        logits, final_state = model(resevoir)
        correct += model.accuracy_function(logits, label_batch)
        total += model.batch_size

    accuracy = correct/total
    return accuracy


def main():
    print("Running preprocessing...")
    train_data, train_labels, test_data, test_labels = get_data("Train_Arabic_Digit.txt", "Test_Arabic_Digit.txt")
    print("Preprocessing complete.")
    model = Model()

    start = time.time()
    epochs = 25
    for idx in range(epochs):
        loss = train(model, train_data, train_labels)
        progress = idx/epochs
        sys.stdout.write('\r')
        sys.stdout.write("[{:20}] {:.2f}% :  loss={:.2f}".format(\
          	'='*int(20*progress), 100*progress, loss))
        sys.stdout.flush()
    print()
    duration = (time.time() - start)/60
    accuracy = test(model, test_data, test_labels)
    print("accuracy={}%, train time={:.2f} minutes".format(int(accuracy * 100), duration))


if __name__ == '__main__':
    main()
