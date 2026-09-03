

import numpy as np


def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
 
    ones = np.ones((X.shape[0], 1))
    X_with_bias = np.hstack([ones, X])  

    weights = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    return weights  


def fit_gradient_descent(X: np.ndarray, y: np.ndarray, lr: float = 0.01, n_iterations: int = 1000):
    """
    Solve the SAME linear regression problem iteratively, to compare
    against fit_normal_equation's exact answer.
    """
    ones = np.ones((X.shape[0], 1))
    X_with_bias = np.hstack([ones, X])
    n_samples, n_features = X_with_bias.shape

    weights = np.zeros(n_features)  
                                     
                                    
    loss_history = []                 
                                       
                                       

    for i in range(n_iterations):
        predictions = X_with_bias @ weights
        errors = predictions - y

      
        gradient = (2 / n_samples) * (X_with_bias.T @ errors)

        weights -= lr * gradient  

        loss = np.mean(errors ** 2)
        loss_history.append(loss)

    return weights, loss_history
