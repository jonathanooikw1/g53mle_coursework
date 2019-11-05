import matplotlib.pyplot as plt
epoch = []
training_loss = []
testing_loss = []

f = open("Results.txt", "r")
for row in f:
    data = row.split()
    epoch.append(int(data[0]))
    training_loss.append(float(data[1]))
    testing_loss.append(float(data[2]))
f.close()
plt.plot(epoch, training_loss, label="Training Loss")
plt.plot(epoch, testing_loss, label="Test Loss")
plt.legend()
plt.show()