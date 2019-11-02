# This python file loads the model and generates and saves the results
import numpy as np

# This function calculates the f-measure for a single set of precision and recall
def calc_f_measure(precision, recall):
    b = 1
    try:
        f_measure = precision * recall * (1 + pow(b, 2)) / (pow(b, 2) * precision + recall)
    except ZeroDivisionError:
        f_measure = 0
    return f_measure


# This function takes in the 5 sets of precision and recall
# and returns the average f-measure
def calc_avg_f_measure(p1, r1, p2, r2, p3, r3, p4, r4, p5, r5):
    total_f_measure = calc_f_measure(p1, r1) + calc_f_measure(p2, r2) + calc_f_measure(p3, r3) \
          + calc_f_measure(p4, r4) + calc_f_measure(p5, r5)
    return total_f_measure/5


# This function takes in the predictions and label for one AU
# and calculates the precision and recall
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


# This function takes in the predictions and labels for 5 AUs
# And calculates the precision and recall for each class
def calc_metrics_2d_array(predictions, actual):
    # Splits the predictions into columns as each column represents on AU
    class_1_precision, class_1_recall = calc_metrics_1d_array(predictions[:, 0], actual[:, 0])
    class_2_precision, class_2_recall = calc_metrics_1d_array(predictions[:, 1], actual[:, 1])
    class_3_precision, class_3_recall = calc_metrics_1d_array(predictions[:, 2], actual[:, 2])
    class_4_precision, class_4_recall = calc_metrics_1d_array(predictions[:, 3], actual[:, 3])
    class_5_precision, class_5_recall = calc_metrics_1d_array(predictions[:, 4], actual[:, 4])
    class_1_fmeasure = calc_f_measure(class_1_precision, class_1_recall)
    class_2_fmeasure = calc_f_measure(class_2_precision, class_2_recall)
    class_3_fmeasure = calc_f_measure(class_3_precision, class_3_recall)
    class_4_fmeasure = calc_f_measure(class_4_precision, class_4_recall)
    class_5_fmeasure = calc_f_measure(class_5_precision, class_5_recall)
    results = np.array([class_1_precision, class_1_recall, class_1_fmeasure,
                        class_2_precision, class_2_recall, class_2_fmeasure,
                        class_3_precision, class_3_recall, class_3_fmeasure,
                        class_4_precision, class_4_recall, class_4_fmeasure,
                        class_5_precision, class_5_recall, class_5_fmeasure])
    print("Class one precision:", format(class_1_precision*100, '.2f') + "%", "Class one recall",
          format(class_1_recall*100, '.2f') + "%")
    print("Class two precision:", format(class_2_precision*100, '.2f') + "%", "Class two recall",
          format(class_2_recall*100, '.2f') + "%")
    print("Class three precision:", format(class_3_precision*100, '.2f') + "%", "Class three recall",
          format(class_3_recall*100, '.2f') + "%")
    print("Class four precision:", format(class_4_precision*100, '.2f') + "%", "Class four recall",
          format(class_4_recall*100, '.2f') + "%")
    print("Class five precision:", format(class_5_precision*100, '.2f') + "%", "Class five recall",
          format(class_5_recall*100, '.2f') + "%")

    avg_f_measure = calc_avg_f_measure(class_1_precision, class_1_recall, class_2_precision, class_2_recall,
                                       class_3_precision, class_3_recall, class_4_precision, class_4_recall,
                                       class_5_precision, class_5_recall)
    print("Average F-Measure:", format(avg_f_measure, '.2f'))
    return results


def cut_off(predictions, cutoff):
    for i in range(len(predictions)):
        for ii in range(len(predictions[i])):
            if predictions[i][ii] > cutoff:
                predictions[i][ii] = 1
            else:
                predictions[i][ii] = 0
    return predictions


def save_results(predictions, actual, epoch):
    results = calc_metrics_2d_array(cut_off(predictions, 0.5), actual)
    results_reformatted = str(epoch)
    for result in results:
        results_reformatted += (" " + str(result))
    f = open("Results.txt", "a+")
    f.write(results_reformatted)
    f.write("\n")
    f.close()


def save_losses(train_loss, test_loss, epoch):
    f = open("Loss.txt", "a+")
    loss = str(epoch) + " " + str(train_loss) + " " + str(test_loss) + "\n"
    f.write(loss)
    f.close()
