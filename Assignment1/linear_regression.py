import numpy as np

class LinearRegression():

    def __init__(self, learning_rate=0.001, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = [] 

    def fit(self, X, y):

        self.weights = 0
        self.bias = 0
        
        for _ in range(self.epochs):
            y_pred = self.predict(X)
            grad_w, grad_b = self.compute_gradients(X,y,y_pred)
            self.update_parameters(grad_w,grad_b)
            loss = self._compute_loss(y,y_pred)
            self.losses.append(loss)
            
    def predict(self, X):
        return X*self.weights + self.bias
    
    def _compute_loss(self, y, y_pred):
        # Mean square error as loss function: (1/n) ||(y-y_pred)**2||
        return np.mean((y-y_pred)**2)

    def compute_gradients(self, X, y, y_pred): # X is a vector
        grad_w = (-2/len(y)) * np.dot((y-y_pred),X)
        grad_b = (-2/len(y)) * sum(y-y_pred)
        return grad_w,grad_b

    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate*grad_w
        self.bias -= self.learning_rate*grad_b