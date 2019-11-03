import matplotlib.pyplot as plt
epoch = []
training_loss = []
testing_loss = []

lowest_loss = 100

f = open("Loss.txt", "r")
for row in f:
    data = row.split()
    epoch.append(int(data[0]))
    training_loss.append(float(data[1]))
    testing_loss.append(float(data[2]))
    if float(data[2]) < lowest_loss:
        lowest_loss = float(data[2])
        lowest_loss_epoch = int(data[0])
f.close()
plt.plot(epoch, training_loss, label="Training Loss")
plt.plot(epoch, testing_loss, label="Test Loss")
print(str(lowest_loss_epoch) + " " + str(lowest_loss))
plt.legend()
plt.show()