import torch
import torch.nn as nn
import torch.optim as optim

import datetime

file_data_2016 = open("data/data_2016.csv", "r")
file_data_2017 = open("data/data_2017.csv", "r")
file_data_2018 = open("data/data_2018.csv", "r")
file_data_2019 = open("data/data_2019.csv", "r")
file_data = [file_data_2016, file_data_2017, file_data_2018, file_data_2019]

# get rid of the headers line
data_2016 = file_data_2016.readlines()[1:]
data_2017 = file_data_2017.readlines()[1:]
data_2018 = file_data_2018.readlines()[1:]
data_2019 = file_data_2019.readlines()[1:]

total_data = data_2016 + data_2017 + data_2018 + data_2019

# getting rid of 'everything but the data, demand, and demand forecast
reduced_total_data = [line.strip("\n").replace("/", ",").replace("\"", "").split(",")[1:4] + \
                      line.strip("\n").replace("/", ",").split(",")[5:-2] for line in total_data]

cleaned_total_data = [[int(val) for val in line] for line in reduced_total_data]

for f_data in file_data:
	f_data.close()

# [month, date, day, demand, demand forecast]
for i in range(0, len(cleaned_total_data)):
    cleaned_total_data[i][2] = datetime.date(cleaned_total_data[i][2], cleaned_total_data[i][0], cleaned_total_data[i][
        1]).weekday()  # replacing the (now useless) year with the number corresponding to the day

    cleaned_total_data[i][0] = (cleaned_total_data[i][0] - 1) / 11  # resizing the data
    cleaned_total_data[i][1] = (cleaned_total_data[i][1] - 1) / 30
    cleaned_total_data[i][2] = (cleaned_total_data[i][2]) / 6
    cleaned_total_data[i][3] = (cleaned_total_data[i][3] - 1) / 1300000


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.hiddenLayer = nn.Linear(3,128)  # month (0-11), 0-1 scaled; day of month (0-30) 0-1 scaled; day of week (0-6), 0-1 scaled
        self.outputLayer = nn.Linear(128, 1)  # first output demand, second output est. generation required

    def forward(self, x):
        x = torch.sigmoid(self.hiddenLayer(x))  # hidden layer with activation
        x = torch.sigmoid(self.outputLayer(x))  # output layer with activation

        return x


'''
#Training the network
'''

# days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}

input_data = torch.tensor([line[0:3] for line in cleaned_total_data])  # input data
output_data = torch.tensor([[float(line[3])] for line in cleaned_total_data])  # output data

'''
# get data for time, temperature, time of year in some big city
# set the lower end of output to be 0, higher end to be 1
# the optimal number should be close to 0 (means low electricity consumption)
'''
model = NeuralNetwork()  # create the neural network; 'model'

lr = .01  # set the learning rate
n_epochs = 50  # an epoch is one cycle through all the training data

loss_fn = nn.MSELoss()  # Use mean squared error

optimizer = optim.SGD(model.parameters(), lr=lr)  # use SGD as the optimizer

model.train()  # set the neural network into training mode
print("training...")
for epoch in range(n_epochs):
    for i in range(input_data.shape[0]):  # for each training example
        prediction = model(input_data[i])

        loss = loss_fn(output_data[i], prediction)  # compute the loss (the error)

        # update the weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("Expected:")
print(output_data)
print("Network output:")
print(model(input_data))
model.eval()  # does not update any weights; for inferencing

# to initialize a model w/ configuration

file_data_2020 = open("data/data_2020.csv", "r")
data_2020 = file_data_2020.readlines()[1:]

reduced_data_2020 = [
    line.strip("\n").replace("/", ",").replace("\"", "").split(",")[1:4] + line.strip("\n").replace("/", ",").split(
        ",")[5:-2] for line in data_2020]  # getting rid of 'everything but the data, demand, and demand forecast

cleaned_data_2020 = [[int(val) for val in line] for line in reduced_data_2020]

for i in range(0, len(cleaned_data_2020)):
    cleaned_data_2020[i][2] = datetime.date(cleaned_data_2020[i][2], cleaned_data_2020[i][0], cleaned_data_2020[i][
        1]).weekday()  # replacing the (now useless) year with the number corresponding to the day

    cleaned_data_2020[i][0] = (cleaned_data_2020[i][0] - 1) / 11  # resizing the data
    cleaned_data_2020[i][1] = (cleaned_data_2020[i][1] - 1) / 30
    cleaned_data_2020[i][2] = (cleaned_data_2020[i][2]) / 6
    cleaned_data_2020[i][3] = (cleaned_data_2020[i][3] - 1) / 1300000

fout = open("predictionsVsActual.txt", "w")
fout.write("My prediction, official prediction, actual demand\n")

# [month, date, day, demand, demand forecast]

for i in range(0, len(cleaned_data_2020)):
    s = ""

    s += (str(float(model(torch.tensor(cleaned_data_2020[i][0:3]))[0]) * 1300000) + ", ")
    s += (str(cleaned_data_2020[i][4]) + ", ")
    s += str(cleaned_data_2020[i][3] * 1300000)
    fout.write(s + "\n")
fout.close()
