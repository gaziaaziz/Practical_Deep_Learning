import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionNumpyForm:
    def __init__(self, X, y):
        self.W = np.random.rand(X.shape[1], 1)
        self.b = 0.5
        self.X = X
        self.y = y
        self.lr = 0.1
    
    def forward(self):
        return np.matmul(self.X, self.W) + self.b
    
    def mse_loss(self, y_pred):
        return np.mean((self.y - y_pred)**2)
    
    def mse_grad(self, y_pred):
        return (-2 * (self.y - y_pred))/ y_pred.shape[0]
    
    def backward(self, grad_mse):
        dW = np.matmul(self.X.T, grad_mse)
        db = np.sum(grad_mse)
        self.W -= (self.lr * dW).flatten()
        self.b -= self.lr * db
    
    
X = np.random.randn(400, 2)
np.random.seed(40)
W_true = np.random.rand(X.shape[1])
b_true = 0.5
y = np.matmul(X, W_true) + b_true
y += y * np.random.uniform(-0.1, 0.1, size= y.shape)
model = LinearRegressionNumpyForm(X, y)
epochs = 100
for _ in range(epochs):
    y_pred = model.forward()
    model.mse_loss(y_pred)
    grad_mse = model.mse_grad(y_pred)
    model.backward(grad_mse)
plt.scatter(X, y, s= 5)
plt.scatter(X, y_pred, s= 5, c='r')
plt.scatter(X, (W_true*X + b_true), s= 5, c= 'g')


