

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
   
    
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def log_loss(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
 
    eps = 1e-15   
                   
    y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iterations: int = 1000):

    ones = np.ones((X.shape[0], 1))
    X_with_bias = np.hstack([ones, X])
    n_samples, n_features = X_with_bias.shape

    weights = np.zeros(n_features)
    loss_history = []

    for i in range(n_iterations):
        z = X_with_bias @ weights
        predictions = sigmoid(z)

     
        errors = predictions - y
        gradient = (1 / n_samples) * (X_with_bias.T @ errors)

        weights -= lr * gradient

        loss_history.append(log_loss(y, predictions))

    return weights, loss_history


def predict_proba(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
  
    ones = np.ones((X.shape[0], 1))
    X_with_bias = np.hstack([ones, X])
    return sigmoid(X_with_bias @ weights)


def predict(X: np.ndarray, weights: np.ndarray, threshold: float = 0.5) -> np.ndarray:
   
    return (predict_proba(X, weights) >= threshold).astype(int)
