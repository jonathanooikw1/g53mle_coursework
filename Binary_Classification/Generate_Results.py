# This python file loads the model and generates and saves the results
import numpy as np

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
    avg_f_measure = calc_avg_f_measure(class_1_precision, class_1_recall)
    difference = np.square(actual - predictions)
    accuracy = 100 - np.sum(difference) / (len(actual) * 2) * 100
    return class_1_precision, class_1_recall, avg_f_measure, accuracy


def cut_off(predictions, cutoff):
    for i in range(len(predictions)):
        for ii in range(len(predictions[i])):
            if predictions[i][ii] > cutoff:
                predictions[i][ii] = 1
            else:
                predictions[i][ii] = 0
    return predictions


def save_results(predictions, actual, epoch):
    class_1_precision, class_1_recall, avg_f_measure, accuracy = calc_metrics_2d_array(cut_off(predictions, 0.5), actual)
    results = str(epoch) + " " + str(class_1_precision) + " " + str(class_1_recall) + " " + str(avg_f_measure) \
              + " " + str(accuracy)
    f = open("Results.txt", "a+")
    f.write(results)
    f.write("\n")
    f.close()
    print("Class one precision:", format(class_1_precision, '.2f') + "%", "Class one recall",
          format(class_1_recall, '.2f') + "%")
    print("Average F-Measure:", format(avg_f_measure, '.2f'))


