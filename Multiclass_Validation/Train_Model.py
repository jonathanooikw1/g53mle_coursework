import time
import os
import warnings
import numpy as np
import tensorflow as tf
import Multiclass_Validation.Generate_Results as results
import Multiclass_Validation.Model as model
data_index = 0


def generate_batch(batch_size):
    global data_index
    batch_features = np.ndarray(shape=(batch_size, n_input), dtype=np.float32)
    batch_labels = np.ndarray(shape=(batch_size, n_output), dtype=np.intc)
    for i in range(batch_size):
        batch_features[i] = X_train[data_index]
        batch_labels[i] = y_train[data_index]
        data_index = (data_index + 1) % len(X_train)
    return batch_features, batch_labels


def multilayer_perceptron(input_d):
    # Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    # Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    # Task of neurons of second hidden layer
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w3), b3))
    # Task of neurons of output layer
    out_layer = tf.nn.softmax(tf.add(tf.matmul(layer_3, w4), b4))
    return out_layer


# Network parameters
n_input = 98
n_hidden1 = 64
n_hidden2 = 64
n_hidden3 = 64
n_output = 5

# Dataset
X_test = np.loadtxt('X_test.txt')
X_train = np.loadtxt('X_train.txt')
y_test = np.loadtxt('y_test.txt')
y_train = np.loadtxt('y_train.txt')

# Learning parameters
learning_constant = 0.2
number_epochs = 20000
batch_size = 32
num_batch = int(len(X_train)/batch_size)

# Placeholders
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

b1, b2, b3, b4, w1, w2, w3, w4, checkpoint = model.load_model(n_input, n_hidden1, n_hidden2, n_hidden3, n_output)
neural_network = multilayer_perceptron(X)

# Define loss and optimizer
# loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))
cross_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(Y, neural_network))
optimizer = tf.train.AdamOptimizer().minimize(cross_loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    start = time.time()
    for epoch in range(int(checkpoint), number_epochs):
        for batch_counter in range(num_batch):
            batch_features, batch_labels = generate_batch(batch_size)
            sess.run(optimizer, feed_dict={X: batch_features, Y: batch_labels})
        if epoch % 50 == 0:
            train_loss = sess.run(cross_loss, feed_dict={X: X_train, Y: y_train})
            test_loss = sess.run(cross_loss, feed_dict={X: X_test, Y: y_test})
            results.save_losses(train_loss, test_loss, epoch)
            time_taken = time.time() - start
            start = time.time()
            print("Epoch: ", epoch, "Test Loss: ", test_loss, " Time taken: ", format(time_taken, '.2f'))
            results.save_results(sess.run(neural_network, feed_dict={X: X_test}), y_test, epoch)
            model.save_model(epoch, sess.run(b1), sess.run(b2), sess.run(b3), sess.run(b4),
                       sess.run(w1), sess.run(w2), sess.run(w3), sess.run(w4))
