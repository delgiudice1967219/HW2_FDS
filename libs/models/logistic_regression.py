import numpy as np
from libs.math import sigmoid

class LogisticRegression:
    def __init__(self, num_features : int):
        self.parameters = np.random.normal(0, 0.01, num_features)
        
    def predict(self, x:np.array) -> np.array:
        """
        Method to compute the predictions for the input features.

        Args:
            x: it's the input data matrix.

        Returns:
            preds: the predictions of the input features.
        """
        thetaX = np.dot(x, self.parameters)
        predicted_probs = sigmoid(thetaX)
        preds = [1 if predicted_prob >= 0.5 else 0 for predicted_prob in predicted_probs]
        return preds
    
    @staticmethod
    def likelihood(preds, y : np.array) -> np.array:
        """
        Function to compute the log likelihood of the model parameters according to data x and label y.

        Args:
            preds: the predicted labels.
            y: the label array.

        Returns:
            log_l: the log likelihood of the model parameters according to data x and label y.
        """
        log_l = 0
        for i in range(len(y)):
            log_l += y*np.log(preds) + (1-y) * np.log(1-preds)
        log_l = log_l / len(y)
        return log_l
    
    def update_theta(self, gradient: np.array, lr : float = 0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the gradient of the log likelihood.
            lr: the learning rate.

        Returns:
            None
        """
        self.parameters = self.parameters + lr * gradient
        pass
        
    @staticmethod
    def compute_gradient(x : np.array, y: np.array, preds: np.array) -> np.array:
        """
        Function to compute the gradient of the log likelihood.

        Args:
            x: it's the input data matrix.
            y: the label array.
            preds: the predictions of the input features.

        Returns:
            gradient: the gradient of the log likelihood.
        """
        gradient = 0
        for i in range(len(y)):
            for j in range(len(x)):
                gradient += (y[i] - preds[i]) * x[i, j]
        gradient = gradient / len(y)
        return gradient

