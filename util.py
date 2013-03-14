import numpy as np

def softmax(X, W):
    """exp(X*W) normalized."""
    [n, m] = X.shape
    A = X.dot(W)
    A = np.exp(A - A.max(axis=1).reshape(n,1))
    A = A / A.sum(axis=1).reshape(n,1)

    return A

def vec_to_mat(t, K):
    """Convert a vector of prediction or labels into a 1-of-K matrix
    representation."""
    if isinstance(t, list):
        ncases = len(t)
    else:   # t is a numpy array
        ncases = t.size

    T = np.zeros((ncases, K), dtype=np.int)
    for i in range(0, ncases):
        k = t[i]
        if isinstance(k, np.ndarray):
            T[i][k[0]] = 1
        else:
            T[i][k] = 1

    return T

def loss_vec_mat(t, T):
    """Compute the loss of prediction vector t, given the ground truth matrix
    T, in a 1-of-K representation."""
    [n, K] = T.shape
    loss = 0
    for i in range(0, n):
        if not T[i][t[i]]:
            loss += 1

    return loss

def square_loss(Y, T):
    """Compute element-wise square error of the prediction matrix Y and ground
    truth matrix T."""
    return ((Y - T) * (Y - T)).sum() / 2


class ActivationTypeError(Exception):
    """Not supported activation function type."""
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        if isinstance(self.msg, str):
            return self.msg
        else:
            return repr(self.msg)

class OutputTypeError(Exception):
    """Not supported output function type."""
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        if isinstance(self.msg, str):
            return self.msg
        else:
            return repr(self.msg)

class ConfigFormatError(Exception):
    """Configuration file format error."""
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class LayerSpec:
    """A class for the type of activation function of a layer"""
    ACT_SIGMOID = 'sigmoid'
    ACT_TANH    = 'tanh'
    ACT_RELU    = 'relu'

    TYPE_SIGMOID = 0
    TYPE_TANH    = 1
    TYPE_RELU    = 2

    def __init__(self, out_dim, activation):
        self.lookup = { LayerSpec.ACT_SIGMOID : LayerSpec.TYPE_SIGMOID,
                LayerSpec.ACT_TANH : LayerSpec.TYPE_TANH, 
                LayerSpec.ACT_RELU : LayerSpec.TYPE_RELU }
        if activation.lower() not in self.lookup:
            raise ActivationTypeError(activation)
        self.act_type = self.lookup[activation]
        self.out_dim = out_dim

class OutputSpec:
    """A class for the type of activation of the output layer"""
    ACT_LINEAR  = 'linear'
    ACT_SOFTMAX = 'softmax'

    TYPE_LINEAR  = 0
    TYPE_SOFTMAX = 1

    def __init__(self, out_dim, output_type):
        self.lookup = { OutputSpec.ACT_LINEAR : OutputSpec.TYPE_LINEAR, 
                OutputSpec.ACT_SOFTMAX : OutputSpec.TYPE_SOFTMAX }
        if output_type.lower() not in self.lookup:
            raise OutputTypeError(output_type)
        self.output_type = self.lookup[output_type]
        self.out_dim = out_dim

class Config:
    """A class for configuration of the neural net."""
    
    # section names
    SEC_DATA = 'data'
    SEC_PARAMETERS = 'parameters'
    SEC_LAYER = 'layer'
    SEC_OUTPUT = 'output'

    # attribute names
    DATA_TRAIN_FILE = 'train_data_file'
    DATA_VAL_FILE = 'val_data_file'
    DATA_TEST_FILE = 'test_data_file'
    DATA_OUTPUT_DIR = 'output_dir'

    PAR_LEARN_RATE = 'learn_rate'
    PAR_INIT_SCALE = 'init_scale'
    PAR_MOMENTUM = 'momentum'
    PAR_WEIGHT_DECAY = 'weight_decay'
    PAR_MINIBATCH_SIZE = 'minibatch_size'
    PAR_NUM_EPOCHS = 'num_epochs'
    PAR_EPOCH_TO_DISPLAY = 'epoch_to_display'
    PAR_EPOCH_TO_SAVE = 'epoch_to_save'
    PAR_RANDOM_SEED = 'random_seed'

    LAYER_TYPE = 'type'
    LAYER_OUT_DIM = 'out_dim'

    OUTPUT_TYPE = 'type'
    OUTPUT_OUT_DIM = 'out_dim'


    def __init__(self, file_name):
        """Set default option values. Read configurations from a text file."""
        # data files
        self.train_data_file = ''
        self.val_data_file = ''
        self.test_data_file = ''
        self.is_val = False
        self.is_test = False
        self.output_dir = ''
        self.output_filename_pattern = 'm%d'
        self.is_output = False

        # parameters for training
        self.learn_rate = 0.001
        self.init_scale = 0.001
        self.momentum = 0.9
        self.weight_decay = 0
        self.minibatch_size = 100
        self.epoch_to_display = 10
        self.epoch_to_save = 100
        self.num_epochs = 1000
        self.random_seed = None

        # information for each layer, read from the file
        self.num_layers = 0

        # default output layer
        self.output = OutputSpec(10, OutputSpec.ACT_LINEAR)

        self._parse_cfg_file(file_name)

    def _parse_cfg_file(self, file_name):
        """Parse a configuration file."""
        import ConfigParser

        cfg = ConfigParser.ConfigParser()
        cfg.read(file_name)
        # make sure the config file is in the right format
        self._check_cfg_sections(cfg)
        
        # starts parsing
        self.num_layers = len(cfg.sections()) - 3
        self.layer = []

        # parse data section
        if cfg.has_option(Config.SEC_DATA, Config.DATA_TRAIN_FILE):
            self.train_data_file = cfg.get(Config.SEC_DATA, Config.DATA_TRAIN_FILE)
        else:
            raise ConfigFormatError('No training data.')

        if cfg.has_option(Config.SEC_DATA, Config.DATA_VAL_FILE):
            self.val_data_file = cfg.get(Config.SEC_DATA, Config.DATA_VAL_FILE)
            self.is_val = True

        if cfg.has_option(Config.SEC_DATA, Config.DATA_TEST_FILE):
            self.test_data_file = cfg.get(Config.SEC_DATA, Config.DATA_TEST_FILE)
            self.is_test = True
            
        if cfg.has_option(Config.SEC_DATA, Config.DATA_OUTPUT_DIR):
            self.output_dir = cfg.get(Config.SEC_DATA, Config.DATA_OUTPUT_DIR)
            self.is_output = True

        # parse parameter section
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_LEARN_RATE):
            self.learn_rate = float(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_LEARN_RATE))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_INIT_SCALE):
            self.init_scale = float(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_INIT_SCALE))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_MOMENTUM):
            self.momentum = float(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_MOMENTUM))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_WEIGHT_DECAY):
            self.weight_decay = float(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_WEIGHT_DECAY))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_MINIBATCH_SIZE):
            self.minibatch_size = int(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_MINIBATCH_SIZE))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_NUM_EPOCHS):
            self.num_epochs = int(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_NUM_EPOCHS))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_EPOCH_TO_DISPLAY):
            self.epoch_to_display = int(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_EPOCH_TO_DISPLAY))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_EPOCH_TO_SAVE):
            self.epoch_to_save = int(cfg.get(Config.SEC_PARAMETERS, 
                Config.PAR_EPOCH_TO_SAVE))
        if cfg.has_option(Config.SEC_PARAMETERS, Config.PAR_RANDOM_SEED):
            self.random_seed = int(cfg.get(Config.SEC_PARAMETERS,
                Config.PAR_RANDOM_SEED))

        # parse layer specifications
        for i in range(1, self.num_layers + 1):
            layer_name = Config.SEC_LAYER + str(i)
            if cfg.has_option(layer_name, Config.LAYER_TYPE) and \
                    cfg.has_option(layer_name, Config.LAYER_OUT_DIM):
                self.layer.append(LayerSpec(
                        int(cfg.get(layer_name, Config.LAYER_OUT_DIM)),
                        cfg.get(layer_name, Config.LAYER_TYPE)))
            else:
                raise ConfigFormatError('Incomplete layer: ' + layer_name)

        # parse output section
        if cfg.has_option(Config.SEC_OUTPUT, Config.OUTPUT_TYPE) and \
                cfg.has_option(Config.SEC_OUTPUT, Config.OUTPUT_OUT_DIM):
            self.output = OutputSpec(
                    int(cfg.get(Config.SEC_OUTPUT, Config.OUTPUT_OUT_DIM)),
                    cfg.get(Config.SEC_OUTPUT, Config.OUTPUT_TYPE))
        else:
            raise ConfigFormatError('Incomplete output layer.')

    def _check_cfg_sections(self, cfg):
        """The config parser cfg should already have the config file read in.
        """
        secs = cfg.sections()

        n_data_sec = 0
        n_par_sec = 0
        n_output_sec = 0
        current_layer = 0

        for section in secs:
            if section == Config.SEC_DATA:
                n_data_sec += 1
                if n_data_sec > 1:
                    raise ConfigFormatError('Multiple data sections.')
            elif section == Config.SEC_PARAMETERS:
                n_par_sec += 1
                if n_par_sec > 1:
                    raise ConfigFormatError('Multiple parameter sections.')
            elif section == Config.SEC_OUTPUT:
                n_output_sec += 1
                if n_output_sec > 1:
                    raise ConfigFormatError('Multiple output sections.')
            elif section.startswith(Config.SEC_LAYER):
                n_layer = int(section[5:])
                if not n_layer == current_layer + 1:
                    raise ConfigFormatError('Repeated/skiped layer definition.')
                current_layer += 1
            else:
                raise ConfigFormatError('Unknown section name.')

        return True

    def display(self):
        """For debug use only."""

        # data section
        print '[' + Config.SEC_DATA + ']'
        print Config.DATA_TRAIN_FILE +  '=' + self.train_data_file
        print Config.DATA_VAL_FILE +    '=' + self.val_data_file
        print 'is_val=' + str(self.is_val)
        print Config.DATA_TEST_FILE +   '=' + self.test_data_file
        print 'is_test=' + str(self.is_test)
        print Config.DATA_OUTPUT_DIR +  '=' + self.output_dir
        print 'is_output=' + str(self.is_output)
        print '\n'

        # parameter section
        print '[' + Config.SEC_PARAMETERS + ']'
        print Config.PAR_LEARN_RATE +       '=' + str(self.learn_rate)
        print Config.PAR_INIT_SCALE +       '=' + str(self.init_scale)
        print Config.PAR_MOMENTUM +         '=' + str(self.momentum)
        print Config.PAR_WEIGHT_DECAY +     '=' + str(self.weight_decay)
        print Config.PAR_MINIBATCH_SIZE +   '=' + str(self.minibatch_size)
        print Config.PAR_NUM_EPOCHS +       '=' + str(self.num_epochs)
        print Config.PAR_EPOCH_TO_DISPLAY + '=' + str(self.epoch_to_display)
        print Config.PAR_EPOCH_TO_SAVE +    '=' + str(self.epoch_to_save)
        print '\n'

        # layer sections
        for i in range(0, self.num_layers):
            print '[' + Config.SEC_LAYER + str(i+1) + ']'
            print Config.LAYER_TYPE +    '=' + str(self.layer[i].act_type)
            print Config.LAYER_OUT_DIM + '=' + str(self.layer[i].out_dim)
            print '\n'

        # output section
        print '[' + Config.SEC_OUTPUT + ']'
        print Config.OUTPUT_TYPE +    '=' + str(self.output.output_type)
        print Config.OUTPUT_OUT_DIM + '=' + str(self.output.out_dim)

