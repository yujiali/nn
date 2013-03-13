import numpy as np
import util

class BaseLayer:
    """Base class for layers in a neural net."""
    def __init__(self, in_dim, out_dim, act_type):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act_type = act_type

    def init_weight(self, init_scale):
        """Initialize the weights to small normally distributed numbers"""
        self.W = init_scale * np.random.randn(self.in_dim, self.out_dim)
        self.Winc = self.W * 0


class Layer(BaseLayer):
    """A layer in the neural network."""
    def forward(self, Xbelow):
        """Take the input from the layer below and compute the output, and
        store the activation in this layer."""

        self.Xbelow = Xbelow
        self.A = self.W.dot(Xbelow)

        if self.act_type == util.LayerSpec.TYPE_SIGMOID:
            self.Xabove = 1/(1 + np.exp(-self.A))
        elif self.act_type == util.LayerSpec.TYPE_TANH:
            expA = np.exp(self.A)
            iexpA = 1 / expA
            self.Xabove = (expA - iexpA) / (expA + iexpA)
        else:       # self.act_type == LayerSpec.TYPE_RELU:
            self.Xabove = np.maximum(self.A, 0)

        return self.Xabove

    def backprop(self, dLdXabove):
        """Compute the gradients at this layer with the input from the layer
        above."""

        if self.act_type == util.LayerSpec.TYPE_SIGMOID:
            self.dXdA = self.Xabove * (1 - self.Xabove)
        elif self.act_type == util.LayerSpec.TYPE_TANH:
            self.dXdA = 1 - self.Xabove * self.Xabove
        else:   # self.act_type == LayerSpec.TYPE_RELU:
            self.dXdA = self.A > 0

        g = self.dLdXaboe * self.dXdA

        self.dLdXbelow = g.dot(self.W.T)
        self.dLdW = self.Xbelow.T.dot(g)

        # [TODO] update W

        return self.dLdXbelow
        
    

class OutputLayer(BaseLayer):
    """The output layer."""
    def forward(self, Xtop):
        self.Xtop = Xtop
        self.Y = util.softmax(Xtop, self.W)

    def loss(self, T):
        self.T = T
        if self.act_type == util.OutputSpec.TYPE_LINEAR:
            self.loss = ((self.Y - T) * (self.Y - T)).sum() / 2
        else: # self.act_type == OutputSpec.TYPE_SOFTMAX:
            pred = self.Y.argmax(axis=1)
            self.loss = util.loss_vec_mat(pred, T)

    def backprop(self):
        self.dLdW = self.Xtop.T.dot(self.Y - T)
        self.dLdXtop = (self.Y - T).dot(self.W.T)

        # [TODO] update W

        return self.dLdXtop


