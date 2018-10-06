from softmax import *
import numpy as np

N = 100000
D = 1000
C = 10
reg = 0.1
X = np.random.randn(N, D)
W = np.random.randn(D, C)
y = np.random.randint(0, C, N)

loss, dW = softmax_loss_naive(W, X, y, reg)
loss2, dW2 = softmax_loss_vectorized(W, X, y, reg)
print(loss, loss2)