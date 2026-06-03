import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(y)
    w = np.zeros(X.shape[1])
    b = 0.0
    loss = 0.0
    
    for epochs in range(steps):
        z = X @ w + b
        prob = _sigmoid(z)

        # Binary cross-entropy loss
        epsilon = 1e-15
        prob_safe = np.clip(prob, epsilon, 1 - epsilon)

        loss = -np.mean(
        y * np.log(prob_safe) +
        (1 - y) * np.log(1 - prob_safe)
        )

        # Gradients
        error = prob - y
        
        dloss_dw = (1 / N) * (X.T @ error)
        dloss_db = (1 / N) * np.sum(error)
        
        w = w - lr * dloss_dw
        b = b - lr * dloss_db
        
    return (w, b)