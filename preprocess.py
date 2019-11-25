import tensorflow as tf
import numpy as np

"""
Takes in file paths for the train and test files.

Returns: inputs and labels for training and testing.
train_inputs: [6600, 93, 13]
train_labels: [6600]
test_inputs: [2200, 93, 13]
test_labels: [2200]
"""
def get_data(train_file, test_file):
    #open files
    with open(train_file, 'r') as f:
        train_data = f.read().split("\n")
    with open(test_file, 'r') as f:
        test_data = f.read().split("\n")


    #process training data
    train_inputs = []
    line_num = 0
    block = np.zeros([93, 13])

    for i in range(1, len(train_data)):
        line = train_data[i].split()
        if len(line) == 13: #line isn't blank
             block[line_num] = line
             line_num += 1
        else:
            line_num = 0
            # print(block)
            # print(block.shape)
            train_inputs.append(block)
            block = np.zeros([93, 13])

    train_inputs = np.array([train_inputs])
    train_inputs = np.transpose(train_inputs, [1, 2, 3, 0])

    #process testing data
    test_inputs = []
    line_num = 0
    block = np.zeros([93, 13])

    for i in range(1, len(test_data)):
        line = test_data[i].split()
        if len(line) == 13: #line isn't blank
             block[line_num] = line
             line_num += 1
        else:
            line_num = 0
            test_inputs.append(block)
            block = np.zeros([93, 13])

    test_inputs = np.array([test_inputs])
    test_inputs = np.transpose(test_inputs, [1, 2, 3, 0])

    #create training and testing labels
    train_labels = []
    test_labels = []

    for i in range(6600):
        train_labels.append(int(i // 660))

    for i in range(2200):
        test_labels.append(int(i // 220))
    
    train_labels = tf.one_hot(train_labels, 10)
    test_labels = tf.one_hot(test_labels, 10)

    return train_inputs, train_labels, test_inputs, test_labels

def main():
    get_data('Train_Arabic_Digit.txt', 'Test_Arabic_Digit.txt')


if __name__ == '__main__':
    main()
