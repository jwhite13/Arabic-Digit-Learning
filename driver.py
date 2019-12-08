import os
import tensorflow as tf
import numpy as np
from preprocess import get_data
from cnn import CNN
import sys
import time

def train(model, train_data, train_labels):
    indices = list(range(len(train_labels)))
    indices = tf.random.shuffle(indices)
    train_data = tf.gather(train_data, indices)
    train_labels = tf.gather(train_labels, indices)

    leng = int(len(train_labels) * 1)
    ttl_loss = 0

    for idx in range(0, leng, model.batch_size):
        data_batch = train_data[idx:idx + model.batch_size]
        label_batch = train_labels[idx:idx + model.batch_size]

        with tf.GradientTape() as tape:
            predictions = model.call(data_batch)
            loss = model.loss(predictions, label_batch)

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
        predictions = model(data_batch)
        correct += model.accuracy_function(predictions, label_batch)
        total += model.batch_size

    accuracy = correct/total
    return accuracy


def main():
    print("Running preprocessing...")
    train_data, train_labels, test_data, test_labels = get_data("Train_Arabic_Digit.txt", "Test_Arabic_Digit.txt")
    print("Preprocessing complete.")
    
    model = CNN(10)

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
