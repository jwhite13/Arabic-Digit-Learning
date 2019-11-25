import tensorflow as tf
import numpy as np

def parse_file(filename):
    out = np.zeros((10000, 93, 13))
    line_num = 0
    idx = 0
    with open(filename, 'r') as data:
        for line in data.readlines():
            split = line.split()
            if len(split) > 0: #line isn't blank
                out[idx, line_num] = np.array(split, dtype=np.float32)
                line_num += 1
            else:
                line_num = 0
                idx += 1

    out = np.expand_dims(out[0:idx], 3)
    return out

"""
Takes in file paths for the train and test files.

Returns: inputs and labels for training and testing.
train_inputs: [6600, 93, 13]
train_labels: [6600]
test_inputs: [2200, 93, 13]
test_labels: [2200]
"""
def get_data(train_file, test_file):
    train_data = parse_file(train_file)
    test_data = parse_file(test_file)
    #create training and testing labels
    train_labels = []
    test_labels = []

    for i in range(6600):
        train_labels.append(int(i // 660))

    for i in range(2200):
        test_labels.append(int(i // 220))
    
    train_labels = tf.one_hot(train_labels, 10)
    test_labels = tf.one_hot(test_labels, 10)

    return train_data, train_labels, test_data, test_labels

def main():
    get_data('Train_Arabic_Digit.txt', 'Test_Arabic_Digit.txt')


if __name__ == '__main__':
    main()
