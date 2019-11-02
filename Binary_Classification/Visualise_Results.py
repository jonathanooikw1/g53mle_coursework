import matplotlib.pyplot as plt

f = open("Results.txt", "r")
epoch = []
precision = []
recall = []
f_measure = []
accuracy = []

for row in f:
    data = row.split()
    epoch.append(int(data[0]))
    precision.append(float(data[1]))
    recall.append(float(data[2]))
    f_measure.append(float(data[3]))
    accuracy.append(float(data[4]))

plt.plot(epoch, precision, label="Precision")
plt.plot(epoch, recall, label="Recall")
plt.plot(epoch, f_measure, label="F-measure")
plt.plot(epoch, accuracy, label="Accuracy")
plt.legend()
plt.show()
