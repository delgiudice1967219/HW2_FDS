import numpy as np

def fit(model, x : np.array, y : np.array, x_val:np.array = None, y_val:np.array = None, lr: float = 0.5, num_steps : int = 500):
    """
    Function to fit the logistic regression model using gradient ascent.

    Args:
        model: the logistic regression model.
        x: it's the input data matrix.
        y: the label array.
        x_val: it's the input data matrix for validation.
        y_val: the label array for validation.
        lr: the learning rate.
        num_steps: the number of iterations.

    Returns:
        history: the values of the log likelihood during the process.
    """

    # Initialize lists to store the log-likelihood values and validation loss values for each iteration
    likelihood_history = np.zeros(num_steps)
    val_loss_history = np.zeros(num_steps)

    # Loop over each iteration
    for it in range(num_steps):

        # Collect all the predicted values for the training set
        preds = model.predict(x)
        
        # Compute the Log likelihood on the training set
        log_l = model.likelihood(preds, y)
        
        # Store the likelihood for the current iteration
        likelihood_history[it] = log_l
        
        # Compute the gradient
        gradient = model.compute_gradient(x, y, preds)
        
        # Update the weights for the current iteration
        model.update_theta(gradient, lr)

        # If a validation set is given as input
        if x_val is not None and y_val is not None:

            # Collect the predicted values for the validation set
            val_preds = model.predict(x_val)
            
            # Compute the negative log-likelihood (loss) on the validation set and store it
            val_loss_history[it] = - model.likelihood(val_preds, y_val)

    return likelihood_history, val_loss_history

