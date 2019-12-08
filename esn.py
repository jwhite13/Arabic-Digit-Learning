import numpy as np
import tensorflow as tf
import scipy as sp

from preprocess import get_data
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

##not in tensorflow -- ask if im even allowed to use these libraries, see what can be converted to tf or np
class Reservoir:

    def __init__ (self, circle=False):
        self.internal_units = 100
        self.noise_level = 0.01
        self.leak = None
        self.radius = 0.99
        self.connectivity = 0.3

        # generate weights in -- what's going on here??
        self.input_weights = None

        # generate weights for reservoir (not trainable)
        self.internal_wts = self.initialize_weights(self.internal_units, self.connectivity, self.radius)


    def initialize_weights(self, internal_units, connectivity, radius):
        # Generate sparse, uniformly distributed weights.
        internal_wts = sp.sparse.rand(internal_units, internal_units, density=connectivity).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        #not sure about this line -- seems like we could just use a normal distribution situation here?
        internal_wts[np.where(internal_wts > 0)] -= 0.5

        # Adjust the spectral radius.
        eigvals, eigvecs = np.linalg.eig(internal_wts)
        eig_max = np.max(np.abs(eigvals))
        internal_wts = internal_wts / (np.abs(eig_max) * radius)

        return internal_wts

    # calculate hidden state h
    def hidden_state(self, inputs, n_drop=0):
        batch_size, num_timesteps, num_vars = inputs.shape
        prev_state = np.zeros([batch_size, self.internal_units])
        hidden_state = np.empty([batch_size, num_timesteps - n_drop, self.internal_units])

        for t in range(num_timesteps):
            curr_timestep = inputs[:, t, :]
            next_state = np.dot(self.internal_wts, prev_state.T) + np.dot(self.internal_wts, curr_timestep.T)#[internal_units, batch_size]

            #add noise -- seems optional
            next_state += np.array(np.random([self.internal_units, batch_size])) * self.noise_level

            #nonlinearity -- we could play around with this, references say tanh
            prev_state = np.tanh(next_state.T) #[batch_size, internal_units]

            #dropout step -- seems to not be used in reference, n_drop = 0
            if t < (self.n_drop - 1): #replace t in next line with t-n_drop
                hidden_state[:, t - n_drop, :] =  prev_state

        return hidden_state

    def reservoir_embedding(self, inputs, pca, n_drop=5, test=False):
        batch_size, num_timesteps, num_vars = inputs.shape

        hidden_state = self.hidden_state(inputs, n_drop)
        print(hidden_state.shape)
        states.reshape([-1, num_vars])

        # if test == True:
        #     reduced_states = pca.fit_transform(states)
        # else:
        #     reduced_states = pca.transform(states)
        # reduced_states = reduced_states.reshape([batch_size, -1, num_timesteps])

        ##??? ridge embedding is the part i don't understand

def accuracy_function(self, labels, logits):
        guesses = np.argmax(logtis, axis=1)
        labels = tf.where(tf.equal(labels, 1))
        labels = labels[:,1]
        accuracy = tf.cast(tf.equal(guesses, labels), dtype=tf.float32)

        return tf.reduce_sum(accuracy)

def train(self, model, train_data, train_labels):
        indices = list(range(len(train_labels)))
        indices = tf.random.shuffle(indices)
        tf.gather(train_data, indices)
        tf.gather(train_labels, indices)

        leng = int(len(train_labels) * 1)
        ttl_loss = 0

        for idx in range(0, 100, model.batch_size):
            data_batch = train_data[idx:idx + model.batch_size]
            label_batch = train_labels[idx:idx + model.batch_size]
            print(data_batch.size) #should be 100x93x13
            print(label_batch.size) #100x10

            train_result = model.reservoir_embedding(train_data, pca)
            # readout.fit(train_result, train_labels)

def test(self, model, test_data, test_labels):
        test_input_result = model.reservoir_embedding(inputs, pca, test=True)
        logits = readout.predict(test_input_result)

        accuracy = self.accuracy_function(test_labels, logits)
        return accuracy

def main():
    print("Running preprocessing...")
    train_data, train_labels, test_data, test_labels = get_data("Train_Arabic_Digit.txt", "Test_Arabic_Digit.txt")
    print("Preprocessing complete.")

    model = Reservoir()
    pca = PCA()
    readout = Ridge(alpha=5)

    epochs = 1
    for i in range(epochs):
        train(model, train_data, train_labels)
    # accuracy = test(model, test_data, test_labels)
    # print(accuracy)

if __name__ == '__main__':
   main()
