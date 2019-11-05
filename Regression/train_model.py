import numpy as np
import tensorflow as tf
import math
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import KFold
import Regression.data as Data
import warnings
warnings.filterwarnings("ignore")
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

#Network parameters
n_hidden1 = 64
n_hidden2 = 64
n_hidden3 = 64
n_input = 98
n_output = 1
#Learning parameters
learning_constant = 0.002
number_epochs = 1000

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

#DEFINING WEIGHTS AND BIASES
#Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1]))
#Biases second hidden layer
b2 = tf.Variable(tf.random_normal([n_hidden2]))
#Biases output layer
b3 = tf.Variable(tf.random_normal([n_hidden3]))
b4 = tf.Variable(tf.random_normal([n_output]))

#Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
#Weights connecting first hidden layer with second hidden layer
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
#Weights connecting second hidden layer with output layer
w3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]))
w4 = tf.Variable(tf.random_normal([n_hidden3, n_output]))


#The incoming data given to the
#network is input_d
def multilayer_perceptron(input_d):
    #Task of neurons of first hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    #Task of neurons of second hidden layer
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    #Task of neurons of output layer
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w3), b3))
    #Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_3, w4),b4)
    
    return out_layer

#Create model
neural_network = multilayer_perceptron(X)
#Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

features, labels = Data.obtain_data()
kf = KFold(n_splits=10)
# Code batch size
batch_size = 64
#Create a session
X_train = []
X_test = []
y_train = []
y_test = []
with tf.Session() as sess:
    sess.run(init)
    #Training epoch
    for epoch in range(0, number_epochs, 10):
        counter = 0
        for train_index, test_index in kf.split(features):
            actual_epoch = epoch + counter
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            num_batch = int(len(X_train) / batch_size)
            for batch_counter in range(num_batch):
                batch_features, batch_labels = generate_batch(batch_size)
                sess.run(optimizer, feed_dict={X: batch_features, Y: batch_labels})
            counter = counter + 1
            # Display the epoch
            # Display the epoch
            if actual_epoch % 10 == 0:
                test_mse = loss_op.eval({X: X_test, Y: y_test})
                train_mse = loss_op.eval({X: X_train, Y: y_train})
                print("Epoch:", '%d' % (actual_epoch))
                print("Test MSE:", test_mse)
                print("Train MSE:", train_mse)
                results = str(actual_epoch) + " " + str(train_mse) + " " + str(test_mse) + "\n"
                f = open("Results.txt", "a+")
                f.write(results)
                f.close()

    # Test model
    pred = (neural_network)  # Apply softmax to logits

    # print("Accuracy:", np.square(accuracy.eval({X: X_train, Y: y_train})).mean())
    output=neural_network.eval({X: X_train})
    plt.plot(y_train, 'r', output, 'b')
    plt.ylabel('Train Results')
    plt.show()

    output_test = neural_network.eval({X: X_test})
    plt.plot(y_test, 'r', output_test, 'b')
    plt.ylabel('Test results')
    plt.show()
    # MSE
    print("MSE:", loss_op.eval({X: X_test, Y: y_test}))

    df = DataFrame(output)
    export_csv = df.to_csv ('output.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
