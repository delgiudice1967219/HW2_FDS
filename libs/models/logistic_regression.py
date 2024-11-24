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

        # Compute the dot product between features and parameters
        thetaX = np.dot(x, self.parameters)

        # Basing on the previous dot product compute the predicted labels
        preds = sigmoid(thetaX)

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

        # Add a samll constant to avoid log of 0 and division by 0
        epsilon = 1e-10

        # Compute the likelihood
        log_l = np.sum(y * np.log(preds + epsilon) + (1 - y) * np.log(1 - preds + epsilon))
        
        return log_l / (len(y) + epsilon) 
    
    def update_theta(self, gradient: np.array, lr : float = 0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the gradient of the log likelihood.
            lr: the learning rate.

        Returns:
            None
        """

        # Update the parameters
        self.parameters += (lr * gradient)
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

        # Compute the errors, defined as the diffrence between the true labels and the predicted ones
        errors = y - preds

        # Compute the gradient of the log likelihood, defined as:
        # - The dot product between the transpose of the input data matrix (x.T) and the errors.
        # - Then, divide by the number of samples (len(y)) to average over all data points.
        gradient = np.dot(x.T, errors) / len(y)
        return gradient

