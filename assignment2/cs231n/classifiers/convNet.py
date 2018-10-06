import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - spatial_batchnorm - relu - 2x2 max pool - affine - normal_batchnorm - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 use_batchnorm=False, dropout=0, dtype=np.float32):

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        #                            INITILIZATION                                 #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = np.random.randn(
            num_filters, C, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.randn(
            int(num_filters*H/2*W/2), hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.randn(
            hidden_dim, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)
        if use_batchnorm:
            self.params['gamma1'] = np.ones(num_filters)
            self.params['beta1'] = np.zeros(num_filters)
            self.params['gamma2'] = np.ones(hidden_dim)
            self.params['beta2'] = np.zeros(hidden_dim)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for _ in range(2)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        mode = 'test' if y is None else 'train'

        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode


        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        alpha = 0.1
        csrp1, csrp1_cache = conv_sbn_lrelu_pool_forward(X, W1, b1, gamma1, beta1, self.bn_params[0], conv_param, pool_param, alpha)
        abr1, abr1_cache = affine_bn_lrelu_forward(csrp1, W2, b2, gamma2, beta2, self.bn_params[1], alpha)
        scores, out_cache = affine_forward(abr1, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dp = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(
                                 np.sum(W1 **  2) + np.sum(W2 ** 2) + np.sum(W3 ** 2)
                                       )
        dp, dw3, db3 = affine_backward(dp, out_cache)
        dp, dw2, db2, dgamma2, dbeta2 = affine_bn_lrelu_backward(dp, abr1_cache)
        dp, dw1, db1, dgamma1, dbeta1 = conv_sbn_lrelu_pool_backward(dp, csrp1_cache)
        grads['W1'] = dw1 + self.reg * W1
        grads['W2'] = dw2 + self.reg * W2
        grads['W3'] = dw3 + self.reg * W3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3
        grads['gamma2'] = dgamma2
        grads['gamma1'] = dgamma1
        grads['beta2'] = dbeta2
        grads['beta1'] = dbeta1
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


pass
