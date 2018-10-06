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

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_size=[7],
                 hidden_dim=[100], num_classes=10, weight_scale=1e-3, reg=0.0,
                 use_batchnorm=False, dropout=0, pool_size=2, pool_stride=2, dtype=np.float32):

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.filter_size = filter_size
        self.pool_param = {'pool_height': pool_size, 'pool_width': pool_size, 'stride': pool_stride}
        self.conv_param = {'stride': 1, 'pad': (self.filter_size[-1] - 1) // 2} 
        ############################################################################
        #                            INITILIZATION                                 #
        ############################################################################
        C, H, W = input_dim
        channel = C
        for i in range(len(self.num_filters)):
            self.params['conv_W'+str(i+1)] = np.random.randn(
                self.num_filters[i], channel, filter_size[i], filter_size[i]) * weight_scale
            self.params['conv_b'+str(i+1)] = np.zeros(self.num_filters[i])
            channel = self.num_filters[i]

        fc_inputdim = int(self.num_filters[-1]*H*W/((2*pool_size) ** len(self.num_filters)))

        for i in range(len(self.hidden_dim)):
            self.params['fc_W'+str(i+1)] = np.random.randn(
				   fc_inputdim , self.hidden_dim[i]) * weight_scale
            self.params['fc_b'+str(i+1)] = np.zeros(self.hidden_dim[i])
            fc_inputdim = self.hidden_dim[i]


        self.params['out_W'] = np.random.randn(
				    self.hidden_dim[-1], num_classes) * weight_scale
        self.params['out_b'] = np.zeros(num_classes)
        if use_batchnorm:
            for i in range(len(self.num_filters)):
                self.params['conv_gamma'+str(i+1)] = np.ones(self.num_filters[i])
                self.params['conv_beta'+str(i+1)] = np.zeros(self.num_filters[i])
            for i in range(len(self.hidden_dim)):
                self.params['fc_gamma'+str(i+1)] = np.ones(self.hidden_dim[i])
                self.params['fc_beta'+str(i+1)] = np.zeros(self.hidden_dim[i])
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
            self.bn_params = [{'mode': 'train'} for _ in range(len(self.num_filters) + len(self.hidden_dim))]

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


        '''
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        '''
        # pass conv_param to the forward pass for the convolutional layer
        #filter_size = W1.shape[2]
        conv_param = self.conv_param

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = self.pool_param

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        alpha = 0.1
        conv_cache = []
        fc_cache = []
        inputs = X
        for i in range(len(self.num_filters)):
            csrp, csrp_cache = conv_sbn_lrelu_pool_forward(inputs, self.params['conv_W'+str(i+1)],  self.params['conv_b'+str(i+1)],
                                            self.params['conv_gamma'+str(i+1)], self.params['conv_beta'+str(i+1)], self.bn_params[i], conv_param, pool_param, alpha)
            inputs = csrp
            conv_cache.append(csrp_cache)
        for i in range(len(self.hidden_dim)):
            abr, abr_cache = affine_bn_lrelu_forward(inputs, self.params['fc_W'+str(i+1)], self.params['fc_b'+str(i+1)], 
                                            self.params['fc_gamma'+str(i+1)], self.params['fc_beta'+str(i+1)], self.bn_params[len(self.num_filters)+i], alpha)
            inputs = abr
            fc_cache.append(abr_cache)
        scores, out_cache = affine_forward(inputs, self.params['out_W'], self.params['out_b'])
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
        dp, grads['out_W'], grads['out_b'] = affine_backward(dp, out_cache)
        for i in range(len(self.hidden_dim)-1, -1, -1):
            dp, grads['fc_W'+str(i+1)], grads['fc_b'+str(i+1)], grads['fc_gamma'+str(i+1)], grads['fc_beta'+str(i+1)] = affine_bn_lrelu_backward(dp, fc_cache[i])
            grads['fc_W'+str(i+1)] += self.reg * self.params['fc_W'+str(i+1)]
            loss += 0.5 * self.reg * np.sum(self.params['fc_W'+str(i+1)] ** 2)
        for i in range(len(self.num_filters)-1, -1, -1):
            dp, grads['conv_W'+str(i+1)], grads['conv_b'+str(i+1)], grads['conv_gamma'+str(i+1)], grads['conv_beta'+str(i+1)] = conv_sbn_lrelu_pool_backward(dp, conv_cache[i])
            grads['conv_W'+str(i+1)] += self.reg * self.params['conv_W'+str(i+1)]
            loss += 0.5 * self.reg * np.sum(self.params['conv_W'+str(i+1)] ** 2)

        '''    
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
        '''
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


pass
