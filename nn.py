from layer import Layer, OutputLayer, LayerConfig, LayerStore
import numpy as np
import cPickle as pickle
import util

class Data:
    """A container for data objects. Has three attributes, X, T and K."""
    def __init__(self):
        pass


class NN:
    """A class for general purpose neural networks, trained with
    backpropagation. The type of activation functions, number of hidden layers
    and number of units in each layer, the output function, and other options 
    during training can be configured."""
    def __init__(self):
        pass

    def init_net(self, config):
        """config is an instance of class Config"""
        
        import os

        self.config = config

        if config.is_output and (not os.path.exists(config.output_dir)):
            os.makedirs(config.output_dir)

        self.train_data = self.read_data(config.train_data_file)

        if config.is_val:
            self.val = self.read_data(config.val_data_file)
        if config.is_test:
            self.test = self.read_data(config.test_data_file)

        [num_total_cases, input_dim] = self.train_data.X.shape
        self.num_total_cases = num_total_cases
        self.input_dim = input_dim

        self.num_minibatches = num_total_cases / config.minibatch_size
        if self.num_minibatches < 1:
            self.num_minibatches += 1

        # initialize the network
        self.num_layers = config.num_layers
        self.layer = []
        in_dim = input_dim
        for i in range(0, self.num_layers):
            self.layer.append(Layer(
                in_dim, config.layer[i].out_dim, config.layer[i].act_type))
            in_dim = config.layer[i].out_dim

        self.output = OutputLayer(in_dim, config.output.out_dim,
                config.output.output_type)

        # initialize the weights in every layer
        self._init_weights(config.init_scale, config.random_seed)

    def _init_weights(self, init_scale, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)

        for i in range(0, self.num_layers):
            self.layer[i].init_weight(init_scale)

        self.output.init_weight(init_scale)

    def train(self):
        config = self.config

        layer_config = LayerConfig()
        layer_config.learn_rate = config.learn_rate
        layer_config.momentum = config.momentum
        layer_config.weight_decay = config.weight_decay

        nnstore = NNStore()
        nnstore.init_from_net(self)

        for epoch in range(0, config.num_epochs):
            # shuffle the data cases
            idx = np.random.permutation(self.num_total_cases)
            train_X = self.train_data.X[idx]
            train_T = self.train_data.T[idx]

            loss = 0

            for batch in range(0, self.num_minibatches):
                i_start = batch * config.minibatch_size
                if not batch == self.num_minibatches - 1:
                    i_end = i_start + config.minibatch_size
                else:
                    i_end = self.num_total_cases

                X = train_X[i_start:i_end]
                T = train_T[i_start:i_end]
                Xbelow = X

                # forward pass
                for i in range(0, self.num_layers):
                    Xbelow = self.layer[i].forward(Xbelow)
                self.output.forward(Xbelow)

                # compute loss
                loss += self.output.loss(T)

                # backprop
                dLdXabove = self.output.backprop(layer_config)
                for i in range(self.num_layers-1, -1, -1):
                    dLdXabove = self.layer[i].backprop(dLdXabove, layer_config)

            # statistics
            avg_loss = 1.0 * loss / self.num_total_cases

            if (epoch + 1) % config.epoch_to_display == 0:
                print 'epoch ' + str(epoch + 1) + ', loss = ' + str(avg_loss)

            if (epoch + 1) % config.epoch_to_save == 0:
                nnstore.update_from_net(self)
                nnstore.write(config.output_dir + '/m' + str(epoch + 1) + '.pdata')

    def read_data(self, data_file_name):
        """(data_file_name) --> data
        Read from the specified data file, return a data object, which is an
        object with three attributes, X, T and K. X and T are the data and
        target matrix respectively, and K is the dimensionality of the output.
        Each of X and T is a matrix with N rows, N is the number of data
        cases"""

        f = open(data_file_name)

        data_dict = pickle.load(f)

        f.close()

        X = data_dict['data']
        t = data_dict['labels']
        K = data_dict['K']

        if len(t.shape) == 1 or t.shape[0] == 1 or t.shape[1] == 1:
            T = util.vec_to_mat(t, K)
        else:
            T = t

        data = Data()
        data.X = X
        data.T = T
        data.K = K

        return data

    def save_net(self, model_file_name):
        """Save the current neural net to a file."""
        pass
        
    def display(self):
        print '[' + str(self.output) + ']'
        for i in range(self.num_layers-1, -1, -1):
            print '[' + str(self.layer[i]) + ']'
        print '[input ' + str(self.input_dim) + ']'

        print 'learn_rate : ' + str(self.config.learn_rate)
        print 'init_scale : ' + str(self.config.init_scale)
        print 'momentum : ' + str(self.config.momentum)
        print 'weight_decay : ' + str(self.config.weight_decay)

class NNStore:
    """An object containing all parameters of the neural network, made easy to
    store and load networks."""
    def __init__(self):
        pass

    def init_from_net(self, net):
        """net should be an instance of NN."""
        self.num_layers = net.num_layers
        self.layer = []
        for i in range(0, self.num_layers):
            layer = LayerStore()
            layer.W = net.layer[i].W
            layer.act_type = net.layer[i].act_type
            self.layer.append(layer)

        output = LayerStore()
        output.W = net.output.W
        output.act_type = net.output.act_type

        self.output = output

    def update_from_net(self, net):
        """Update the weights at each layer in a net."""
        for i in range(0, self.num_layers):
            self.layer[i].W = net.layer[i].W
        self.output.W = net.output.W

    def write(self, file_name):
        """Write the net to a file."""
        f = open(file_name, mode='w')
        pickle.dump(self, f)
        f.close()

    def load(self, file_name):
        """Load a net from a file."""
        f = open(file_name)
        nnstore = pickle.load(f)
        f.close()

        self.num_layers = nnstore.num_layers
        self.layer = nnstore.layer
        self.output = nnstore.output

        del nnstore

