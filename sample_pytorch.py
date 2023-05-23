import torch
import torch.nn as nn
import torch.optim as optim

########### Neural network class definition ###################################
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # super initializes a class of the superclass nn.Module

        self.hidden_layer = nn.Linear(2, 2) # input nodes = 2, output = 2
        self.output_layer = nn.Linear(2, 2) # input = 2, output = 2


    def forward(self, x): # the forward pass through the network
        x = torch.sigmoid(self.hidden_layer(x)) # hidden layer with activation
        x = torch.sigmoid(self.output_layer(x)) # output layer with activation

        return x

########### Setup and network training ########################################

input_data = torch.tensor([[.8, .2], [.4, .4]]) # sample input data (2 data points)
output_data = torch.tensor([[.1, .1], [.5, .5]]) # output data

model = NeuralNetwork() # create the neural network

lr = .05 # set the learning rate
n_epochs = 30000 # an epoch is one cycle through all the training data

loss_fn = nn.MSELoss() # Use mean squared error

optimizer = optim.SGD(model.parameters(), lr=lr) # use SGD as the optimizer

model.train() # set the neural network into training mode

for epoch in range(n_epochs):
    for i in range(input_data.shape[0]): # for each training example
        prediction = model(input_data[i])

        loss = loss_fn(output_data[i], prediction) # compute the loss (the error)

        # update the weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print ("Expected:")
print (output_data)
print ("Network output:")
print (model(input_data))
