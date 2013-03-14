if __name__ == "__main__":
    import nn
    import util

    config = util.Config('784.cfg')
    
    net = nn.NN()
    net.init_net(config)
    net.display()
    net.train()

