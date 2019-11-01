import time
import os
import warnings
import numpy as np
import tensorflow as tf
data_index = 0


def calc_f_measure(precision, recall):
    b = 1
    return precision * recall * (1 + pow(b, 2)) / (pow(b, 2) * precision + recall)


def calc_avg_f_measure(p1, r1):
    total_f_measure = calc_f_measure(p1, r1)
    return total_f_measure


def calc_metrics_1d_array(predictions, actual):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and actual[i] == 1:
            true_positive = true_positive + 1
        elif predictions[i] == 1 and actual[i] == 0:
            false_positive = false_positive + 1
        elif predictions[i] == [0] and actual[i] == 1:
            false_negative = false_negative + 1
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall = 0
    return precision, recall


def calc_metrics_2d_array(predictions, actual):
    class_1_precision, class_1_recall = calc_metrics_1d_array(predictions[:, 0], actual[:, 0])
    print("Class one precision:", format(class_1_precision, '.2f') + "%", "Class one recall",
          format(class_1_recall, '.2f') + "%")

    avg_f_measure = calc_avg_f_measure(class_1_precision, class_1_recall)
    print("Average F-Measure:", format(avg_f_measure, '.2f'))


def load_model():
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


def generate_batch(batch_size):
    global data_index
    batch_features = np.ndarray(shape=(batch_size, n_input), dtype=np.float32)
    batch_labels = np.ndarray(shape=(batch_size, n_output), dtype=np.intc)
    for i in range(batch_size):
        batch_features[i] = X_train[data_index]
        batch_labels[i] = y_train[data_index]
        data_index = (data_index + 1) % len(X_train)
    return batch_features, batch_labels


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
n_output = 2

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

b1, b2, b3, b4, w1, w2, w3, w4, checkpoint = load_model()
neural_network = multilayer_perceptron(X)

# Define loss and optimizer
# loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network, Y))
cross_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(Y, neural_network))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(cross_loss)

init = tf.global_variables_initializer()


def cut_off(predictions, cutoff):
    for i in range(len(predictions)):
        for ii in range(len(predictions[i])):
            if predictions[i][ii] > cutoff:
                predictions[i][ii] = 1
            else:
                predictions[i][ii] = 0
    return predictions


with tf.Session() as sess:
    sess.run(init)
    start = time.time()
    for epoch in range(int(checkpoint), number_epochs):
        for batch_counter in range(num_batch):
            batch_features, batch_labels = generate_batch(batch_size)
            sess.run(optimizer, feed_dict={X: batch_features, Y: batch_labels})
        if epoch % 100 == 0:
            c = sess.run(cross_loss, feed_dict={X: X_test, Y: y_test})
            time_taken = time.time() - start
            start = time.time()
            print("Epoch: ", (epoch + 1), "Test Loss: ", c, " Time taken: ", format(time_taken, '.2f'))
            difference = y_test-cut_off(sess.run(neural_network, feed_dict={X: X_test}), 0.5)
            difference = np.square(difference)
            calc_metrics_2d_array(cut_off(sess.run(neural_network, feed_dict={X: X_test}), 0.5), y_test)
            print("Test accurarcy: ", format(100 - np.sum(difference)/(len(y_test)*2)*100, '.2f') + "%")
            save_model(epoch, sess.run(b1), sess.run(b2), sess.run(b3), sess.run(b4),
                       sess.run(w1), sess.run(w2), sess.run(w3), sess.run(w4))
