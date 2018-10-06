import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_samples = X.shape[0]
  for i in range(n_samples):
    a = X[i].reshape((1, -1)).dot(W)
    numerator = np.exp(a - np.max(a)) 
    denmerator = numerator.sum()
    S = numerator / denmerator
    loss += -np.log(S[:, y[i]])
    S[:, y[i]] -= 1
    dW += X[i].reshape((-1, 1)).dot((S.reshape((1, -1))))
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss[0] / n_samples, dW / n_samples


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_samples = X.shape[0]
  A = X.dot(W)
  numerator = np.exp(A - np.max(A, axis=1, keepdims=True))
  denmerator = np.sum(numerator, axis=1, keepdims=True)
  S = numerator / denmerator

  loss += -np.mean(np.log(S[range(n_samples), y])) + 0.5 * reg * np.sum(W ** 2)
  S[range(n_samples), y] -= 1
  dW = X.T.dot(S) / n_samples + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

