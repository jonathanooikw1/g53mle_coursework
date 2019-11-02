import time
import os
import warnings
import numpy as np
import tensorflow as tf


def save_model(epoch, b1, b2, b3, b4, w1, w2, w3, w4):
    try:
        os.mkdir("Models")
    except FileExistsError:
        x = 1 + 1
    f = open(os.path.join("Models", "Versions.txt"), "a+")
    f.write(str(epoch) + "\n")
    np.savetxt(os.path.join("Models", 'b1_' + str(epoch) + '.txt'), b1, fmt='%s')
    np.savetxt(os.path.join("Models", 'b2_' + str(epoch) + '.txt'), b2, fmt='%s')
    np.savetxt(os.path.join("Models", 'b3_' + str(epoch) + '.txt'), b3, fmt='%s')
    np.savetxt(os.path.join("Models", 'b4_' + str(epoch) + '.txt'), b4, fmt='%s')
    np.savetxt(os.path.join("Models", 'w1_' + str(epoch) + '.txt'), w1, fmt='%s')
    np.savetxt(os.path.join("Models", 'w2_' + str(epoch) + '.txt'), w2, fmt='%s')
    np.savetxt(os.path.join("Models", 'w3_' + str(epoch) + '.txt'), w3, fmt='%s')
    np.savetxt(os.path.join("Models", 'w4_' + str(epoch) + '.txt'), w4, fmt='%s')
    f.close()


def load_model(n_input, n_hidden1, n_hidden2, n_hidden3, n_output):
    if os.path.isdir("Models"):
        with open(os.path.join("Models", "Versions.txt"), 'r') as f:
            lines = f.read().splitlines()
            checkpoint = lines[-1]
        print("Loading pretrained model at epoch "+ checkpoint)

        b1 = tf.Variable(np.loadtxt(os.path.join("Models", 'b1_' + checkpoint + '.txt'), dtype='float32'))
        # Biases second hidden layer
        b2 = tf.Variable(np.loadtxt(os.path.join("Models", 'b2_' + checkpoint + '.txt'), dtype='float32'))
        # Biases third hidden layer
        b3 = tf.Variable(np.loadtxt(os.path.join("Models", 'b3_' + checkpoint + '.txt'), dtype='float32'))
        # Biases output layer
        b4 = tf.Variable(np.loadtxt(os.path.join("Models", 'b4_' + checkpoint + '.txt'), dtype='float32'))

        # Weights connecting input layer with first hidden layer
        w1 = tf.Variable(np.loadtxt(os.path.join("Models", 'w1_' + checkpoint + '.txt'), dtype='float32'))
        # Weights connecting first hidden layer with second hidden layer
        w2 = tf.Variable(np.loadtxt(os.path.join("Models", 'w2_' + checkpoint + '.txt'), dtype='float32'))
        # Weights connecting first hidden layer with second hidden layer
        w3 = tf.Variable(np.loadtxt(os.path.join("Models", 'w3_' + checkpoint + '.txt'), dtype='float32'))
        # Weights connecting second hidden layer with output layer
        w4 = tf.Variable(np.loadtxt(os.path.join("Models", 'w4_' + checkpoint + '.txt'), dtype='float32'))
    else:
        print("Randomly initializing model")
        # DEFINING WEIGHTS AND BIASES
        b1 = tf.Variable(tf.random_normal([n_hidden1]))
        # Biases second hidden layer
        b2 = tf.Variable(tf.random_normal([n_hidden2]))
        # Biases third hidden layer
        b3 = tf.Variable(tf.random_normal([n_hidden3]))
        # Biases output layer
        b4 = tf.Variable(tf.random_normal([n_output]))

        # Weights connecting input layer with first hidden layer
        w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
        # Weights connecting first hidden layer with second hidden layer
        w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
        # Weights connecting first hidden layer with second hidden layer
        w3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]))
        # Weights connecting second hidden layer with output layer
        w4 = tf.Variable(tf.random_normal([n_hidden3, n_output]))
        checkpoint = 0
    return b1, b2, b3, b4, w1, w2, w3, w4, checkpoint
