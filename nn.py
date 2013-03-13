from layer import Layer, OutputLayer
import numpy as np

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
            os.mkdirs(config.output_dir)

        self.train = self.read_data(config.train_data_file)

        if config.is_val:
            self.val = self.read_data(config.val_data_file)
        if config.is_test:
            self.test = self.read_data(config.test_data_file)

        [num_total_cases, input_dim] = self.train.shape
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
        # [TODO] add random_seed here
        self._init_weights(config.init_scale)

    def _init_weights(self, init_scale, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)

        for i in range(0, self.num_layers):
            self.layer[i].init_weight(init_scale)

        self.output.init_weight(init_scale)

    def train(self):
        config = self.config

        for epoch in range(0, config.num_epochs):
            # shuffle the data cases
            idx = np.random.permutation(self.num_total_cases)
            train_X = self.train.X[idx]
            train_T = self.train.T[idx]

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
                Y = self.output.forward(Xbelow)

                # compute loss
                loss += self.output.loss(Y, T)

                # backprop
                dLdXabove = self.output.backprop()
                for i in range(self.num_layers-1, -1, -1):
                    dLdXabove = self.layer[i].backprop(dLdXabove)

            # statistics
            avg_loss = loss / self.num_total_cases

            if epoch % self.epoch_to_display == 0:
                print 'epoch ' + str(epoch) + ', loss = ' + avg_loss

    def read_data(self, data_file_name):
        """(data_file_name) --> data
        Read from the specified data file, return a data object, which is an
        object with two attributes, X and T, for data and target respectively,
        each of them is a matrix with N rows, N is the number of data cases"""
        
        # [TODO]


