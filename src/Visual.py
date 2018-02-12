import os
import numbers
import numpy as np
import matplotlib.pyplot as plt

def plot_hidden_weights(weights, size, plot_dir_path, seed = 0):
    """
    Plot hidden weights and save to files

    Arguments:
    weights -- weights of hidden neurons of shape (number of inputs, number of neurons)
    size -- if real then must be in [0.0, 1.0] indicating the percentage of neurons whose
            weights to be plotted; otherwise must be an integer indicating that whole number
    plot_dir_path -- directory to store the resulting files

    Return: None
    """
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path)
    np.random.seed(seed)
    n_neurons = weights.shape[1]
    if isinstance(size, numbers.Integral):
        assert(1 <= size <= n_neurons), "Invalid number of neurons to be plotted"
        n_wanted = size
    elif isinstance(size, numbers.Real):
        assert(0.0 <= size <= 1.0), "Invalid ratio"
        n_wanted = int(size * n_neurons)
    else:
        raise ValueError("Invalid type of size")
    indices = np.random.permutation(np.arange(n_neurons))[:n_wanted]
    for idx in indices:
        fig = plt.figure()
        plt.plot(weights[:,idx])
        plot_file_path = os.path.join(plot_dir_path, "weights_neuron_{}".format(idx))
        plt.savefig(plot_file_path)
        plt.close()

def plot_reconstructed_outputs(X_train, y_train, X_recon, size_per_class, plot_dir_path, seed = 0):
    """
    Plot the original inputs and reconstructed outputs in each class

    Arguments:
    X_train, y_train -- train samples and corresponding targets
    X_recon -- reconstructed outputs, same shape with X_train
    size_per_class -- if real then must be in [0.0, 1.0] indicating the percentage of neurons whose
                      weights to be plotted; otherwise must be an integer indicating that whole number
    plot_dir_path -- directory to save the resulting file
    seed -- numpy random seed

    Return: None
    """
    np.random.seed(seed)
    d = dict([(target, []) for target in y_train])
    for idx, target in enumerate(y_train):
        d[target] += [idx]
    for target, indices in d.items():
        target_dir_path = os.path.join(plot_dir_path, "class_{}".format(target))
        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)
        n_examples = len(indices)
        if isinstance(size_per_class, numbers.Integral):
            assert(1 <= size_per_class <= n_examples), "Invalid number of neurons to be plotted"
            n_wanted = size_per_class
        elif isinstance(size_per_class, numbers.Real):
            assert(0.0 <= size_per_class <= 1.0), "Invalid ratio"
            n_wanted = int(size * n_examples)
        else:
            raise ValueError("Invalid type of size")
        indices = np.random.permutation(np.arange(n_examples))[:n_wanted]
        for idx in indices:
            fig = plt.figure()
            axe1 = fig.add_subplot(211)
            axe1.plot(X_train[idx])
            axe1.set_xlabel("X_train[{}]".format(idx))
            axe2 = fig.add_subplot(212)
            axe2.plot(X_recon[idx])
            axe2.set_xlabel("X_recon[{}]".format(idx))            
            plot_file_path = os.path.join(target_dir_path, "example_{}".format(idx))
            plt.savefig(plot_file_path)
            plt.close()
        

    
