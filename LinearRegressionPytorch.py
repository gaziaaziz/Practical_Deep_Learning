import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

class LinearRegressionV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad= True, dtype= torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad= True, dtype= torch.float))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.weights * X + self.bias


def plot_graphs(X_train, X_test, y_train, y_test, predictions = None):
    plt.figure(figsize= (10, 7))
    plt.scatter(X_train, y_train, c= "b", label= "Training Data")
    plt.scatter(X_test, y_test, c= "g", label= "Testing Data")

    if predictions is not None:
        plt.scatter(X_test, predictions, c= "r", label= "Predictions")
    
    plt.legend(prop= {"size": 14})
    



start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim= 1)
W = 0.7
b = 0.3

y = W*X + b

train_test_split = int(0.8 * len(X))
X_train, y_train = X[:train_test_split], y[:train_test_split]
X_test, y_test = X[train_test_split:], y[train_test_split:]

model_0 = LinearRegressionV1()
mse_loss = nn.MSELoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr = 0.01)


epochs = 100
epoch_count = []
loss_values = []
test_loss_values = []
for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = mse_loss(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #model.eval() is used to evaluate model on testing data. This is important because some settings are not needed in evaluation/testing mode of the model(like dropout or batch norm)
    model_0.eval()
    #use inference mode because it shuts off gradient tracking which is not required in the testing mode
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = mse_loss(test_pred, y_test)
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test_Loss: {test_loss} ")
with torch.inference_mode():
    y_pred = model_0(X_test)

plot_graphs(X_train, X_test, y_train, y_test, y_pred)


class LinearRegressionV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features= 1, out_features= 1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(X)



start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim= 1)
W = 0.7
b = 0.3

y = W*X + b

train_test_split = int(0.8 * len(X))
X_train, y_train = X[:train_test_split], y[:train_test_split]
X_test, y_test = X[train_test_split:], y[train_test_split:]

model_1 = LinearRegressionV2()
mse_loss = nn.MSELoss()
optimizer = torch.optim.SGD(params= model_1.parameters(),lr = 0.01)

torch.manual_seed(42)

epochs = 200
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_1.train()

    y_pred = model_1(X_train) 

    loss = mse_loss(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = mse_loss(test_pred, y_test)
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch {epoch} | Loss {loss} | Test Loss: {test_loss}")

with torch.inference_mode():
    y_pred = model_1(X_test)

plot_graphs(X_train, X_test, y_train, y_test, y_pred)




    
