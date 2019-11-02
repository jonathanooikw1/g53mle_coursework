import time
import os
import warnings
import numpy as np
import tensorflow as tf
data_index = 0
import Binary_Classification.Generate_Results as results
import Binary_Classification.Model as model


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

n_input = 98
n_hidden1 = 64
n_hidden2 = 64
n_hidden3 = 64
n_output = 2

y_test = np.loadtxt('y_test.txt')
X_test = np.loadtxt('X_test.txt')

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

b1, b2, b3, b4, w1, w2, w3, w4, checkpoint = model.load_model(n_input, n_hidden1, n_hidden2, n_hidden3, n_output)
neural_network = multilayer_perceptron(X)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    predictions = sess.run(neural_network, feed_dict={X: X_test})
    class_1_precision, class_1_recall, avg_f_measure, accuracy = \
        results.calc_metrics_2d_array(results.cut_off(predictions, 0.5), y_test)
    print(class_1_precision, class_1_recall, avg_f_measure, accuracy)
