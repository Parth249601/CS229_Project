# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt

def calc_grad(X, Y, theta, reg_lambda=0.01):
    m = X.shape[0]
    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1. / m) * X.T.dot(probs * Y)

    # Regularize (excluding bias term)
    reg = np.concatenate([[0], reg_lambda * theta[1:]])
    grad += reg

    return grad

def calc_grad_no_reg(X, Y, theta):
    """Compute the gradient of the loss without regularization."""
    m = X.shape[0]
    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1. / m) * X.T.dot(probs * Y)
    
    return grad 

def compute_cost(X, Y, theta, reg_lambda=0.01):
    m = len(Y)
    margins = Y * X.dot(theta)
    loss = (1. / m) * np.sum(np.log(1 + np.exp(-margins)))

    # Add L2 regularization (skip theta[0] which is the bias)
    reg = (reg_lambda / 2) * np.sum(theta[1:] ** 2)
    
    return loss + reg


def logistic_regression(X, Y, reg_lambda=0.01):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 0.1
    cost_history = []

    i = 0
    while True and i < 100000:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta, reg_lambda=reg_lambda)
        theta = theta - learning_rate * grad

        # Compute and store cost
        cost = compute_cost(X, Y, theta, reg_lambda=reg_lambda)
        cost_history.append(cost)

        if i % 10000 == 0:
            print(f'Iter {i} | Cost: {cost:.4f}')
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print(f'Converged in {i} iterations')
            break

    # Plot cost after training
    plt.figure()
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    #plt.yscale('log')
    plt.grid(True)
    plt.show()

    return theta



def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    theta_A = logistic_regression(Xa, Ya,1)
    

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    theta_B = logistic_regression(Xb, Yb, 1)

    util.plot(Xa, Ya, theta_A, '../plots/p01_a.png')
    util.plot(Xb, Yb, theta_B, '../plots/p01_b.png')
    



if __name__ == '__main__':
    main()
