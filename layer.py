import numpy as np
import util

class LayerConfig:
    """Configuration of a layer's settings for learning. Containing the
    following attributes
    - learn_rate
    - momentum
    - weight_decay
    - [To be added: sparsity, drop-out, etc.]
    """
    def __init__(self):
        pass

class LayerStore:
    """An object to store a layer's type of activation function, and the
    weight matrix. List of attributes:
    - W
    - act_type
    """
    def __init__(self):
        pass

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
        self.A = Xbelow.dot(self.W)

        if self.act_type == util.LayerSpec.TYPE_SIGMOID:
            self.Xabove = 1/(1 + np.exp(-self.A))
        elif self.act_type == util.LayerSpec.TYPE_TANH:
            expA = np.exp(self.A)
            iexpA = 1 / expA
            self.Xabove = (expA - iexpA) / (expA + iexpA)
        else:       # self.act_type == LayerSpec.TYPE_RELU:
            self.Xabove = np.maximum(self.A, 0)

        return self.Xabove

    def backprop(self, dLdXabove, config):
        """Compute the gradients at this layer with the input from the layer
        above."""

        if self.act_type == util.LayerSpec.TYPE_SIGMOID:
            self.dXdA = self.Xabove * (1 - self.Xabove)
        elif self.act_type == util.LayerSpec.TYPE_TANH:
            self.dXdA = 1 - self.Xabove * self.Xabove
        else:   # self.act_type == LayerSpec.TYPE_RELU:
            self.dXdA = self.A > 0

        self.dLdXabove = dLdXabove
        g = self.dLdXabove * self.dXdA

        self.dLdXbelow = g.dot(self.W.T)
        self.dLdW = self.Xbelow.T.dot(g)

        # update W
        Winc = -config.learn_rate * self.dLdW
        if config.weight_decay > 0:
            Winc -= config.weight_decay * self.W
        if config.momentum > 0:
            Winc += config.momentum * self.Winc

        self.W += Winc
        self.Winc = Winc

        return self.dLdXbelow

    def __str__(self):
        s = 'layer' + str(self.out_dim) + ' '
        if self.act_type == util.LayerSpec.TYPE_SIGMOID:
            s += util.LayerSpec.ACT_SIGMOID
        elif self.act_type == util.LayerSpec.TYPE_TANH:
            s += util.LayerSpec.ACT_TANH
        else: # self.act_type == OutputSpec.TYPE_RELU
            s += util.LayerSpec.ACT_RELU
        return s


class OutputLayer(BaseLayer):
    """The output layer."""
    def forward(self, Xtop):
        """Perform the forward pass, given the top layer output of the net, go
        through the output layer and compute the output."""
        self.Xtop = Xtop
        if self.act_type == util.OutputSpec.TYPE_LINEAR:
            self.Y = Xtop.dot(self.W)
        else:   # self.act_type == util.OutputSpec.TYPE_SOFTMAX
            self.Y = util.softmax(Xtop, self.W)

    def loss(self, T):
        """Compute the loss of the current prediction compared with the given
        ground truth. This function should be called after forward function."""
        self.T = T
        if self.act_type == util.OutputSpec.TYPE_LINEAR:
            loss = ((self.Y - T) * (self.Y - T)).sum() / 2
        else: # self.act_type == OutputSpec.TYPE_SOFTMAX:
            pred = self.Y.argmax(axis=1)
            loss = util.loss_vec_mat(pred, T)

        return loss

    def backprop(self, config, T=None):
        """Backprop through the top layer, using the recorded ground truth
        when calling the loss function or supplied when calling this fucntion.
        The weights of this output layer is updated according to the
        configuration specified in config. The graident of the lower layer
        outputs is returned."""
        if T:
            self.T = T

        self.dLdW = self.Xtop.T.dot(self.Y - self.T)
        self.dLdXtop = (self.Y - self.T).dot(self.W.T)

        # update W
        Winc = -config.learn_rate * self.dLdW
        if config.weight_decay > 0:
            Winc -= config.weight_decay * self.W
        if config.momentum > 0:
            Winc += config.momentum * self.Winc

        self.W += Winc
        self.Winc = Winc

        return self.dLdXtop

    def __str__(self):
        s = 'output ' + str(self.out_dim) + ' '
        if self.act_type == util.OutputSpec.TYPE_LINEAR:
            s += util.OutputSpec.ACT_LINEAR
        else: # self.act_type == OutputSpec.TYPE_SOFTMAX
            s += util.OutputSpec.ACT_SOFTMAX
        return s

