import numpy as np

class LogisticRegression():
    
    def __init__(self, learning_rate=0.01, epochs=200):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
        
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1]) 
        self.bias = 0
        # Gradient descent
        for _ in range(self.epochs):
            lin_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid_function(lin_model)
            grad_w, grad_b = self.compute_gradients(X,y,y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y,y_pred)
            pred_to_class = [1 if _y >= 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y,pred_to_class))
            self.losses.append(loss)
    
    def predict(self, X):
        lin_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return [0 if _y <= 0.5 else 1 for _y in y_pred]

    def sigmoid_function(self, x):
        return 1/(1 + np.exp(-x))
    
    def _compute_loss(self, y, y_pred):
        return np.mean(-y * np.log(y_pred) - (1-y) * np.log(1 - y_pred))

    def compute_gradients (self, X, y, y_pred): 
        grad_w = (1/X.shape[0]) * np.dot(X.transpose(), (y_pred-y))
        grad_b = np.mean(y_pred-y)
        return grad_w, grad_b

    def update_parameters (self, grad_w, grad_b):
        self.weights = self.weights - self.learning_rate * grad_w
        self.bias = self.bias - self.learning_rate * grad_b

    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)
    
