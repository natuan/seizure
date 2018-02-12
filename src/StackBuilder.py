import os
from StackedAutoencoders import *
from Utils import timestr, unit_config_str

class StackBuilder:
    def __init__(self,
                 X,
                 name,
                 unit_type,
                 hidden_activation,
                 n_units,
                 l2_reg_range = None,
                 noise_stddev_range = None,
                 dropout_rate_range = None,
                 cache_dir = "../cache",
                 tf_log_dir = "../tf_logs"):
        """
        Ctor
        
        Params:
        - unit_type: "ordinary", "denoise", "dropout"
        - hidden_activation: "softmax", "relu", "elu"
        - n_units: number of unit autoencoders
        - accepted_reconstruction_loss: if specified and the reconstruction loss <= the value of this param then the unit 
        is considered acceptable; otherwise, the search tries to find units with the lowest reconstruction loss

        Return: None
        """
        assert(n_units is not None), "Invalid n_units"
        self.X = X
        self.name = name
        self.unit_type = unit_type
        self.n_units = n_units
        self.hidden_activation = hidden_activation
        self.l2_reg_range = l2_reg_range
        self.noise_stddev_range = noise_stddev_range
        self.dropout_rate_range = dropout_rate_range
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.tf_log_dir = tf_log_dir
        if not os.path.exists(self.tf_log_dir):
            os.makedirs(self.tf_log_dir)
        self.search_log_file = os.path.join(self.cache_dir, "search_log_{}.txt".format(timestr()))
        self.autoencoder = None

        # Tunable hyper-params for search
        self.n_epochs = 1000
        self.batch_size = 256
        self.checkpoint_steps = 100
        self.seed = 42
        self.epsilon = 1e-08
        self.max_search_trials = 128
        
    def _get_neurons_range(self, n_inputs):
        if self.unit_type == "ordinary":
            min_neurons = 2
            max_neurons = max(min_neurons, n_inputs - 1)
        else:
            min_neurons = n_inputs + 1
            max_neurons = 2 * n_inputs
        return (min_neurons, max_neurons)

    def _get_unit_name(self, prefix, n_inputs, n_neurons, regularizer=None, noise_stddev=None, dropout_rate=None):
        return "{}_inputs{}_neurons{}".format(prefix, n_inputs, n_neurons)
       
    def search_ordinary_unit_helper(self, X, n_neurons, l2_reg_range=(0.001,0.01), accepted_reconstruction_loss = None, n_trials = 0, search_logging = True):
        """
        Recursively search for ordinary autoencoder unit with a given number of hidden neurons

        Params:

        Return: (unit, model_path, reconstruction_loss)
        """
        min_reg, max_reg = l2_reg_range
        if n_trials > self.max_search_trials or min_reg > max_reg:
            return None, None, None
        n_inputs = X.shape[1]
        reg = (min_reg + max_reg) / 2
        if search_logging:
            with open(self.search_log_file, "a") as f:
                f.write("\t>> Constructing and training unit of {} neurons, regularizer {}, trial {}...\n".format(n_neurons, reg, n_trials))
        name = unit_config_str("ord", n_inputs, n_neurons, hidden_activation=self.hidden_activation, regularizer_value=reg)
        model_path = os.path.join(self.cache_dir, name, "{}.model".format(name))
        unit = UnitAutoencoder(name=name,
                               n_inputs=n_inputs,
                               n_neurons=n_neurons,
                               hidden_activation=self.hidden_activation,
                               regularizer=tf.contrib.layers.l2_regularizer(reg),
                               tf_log_dir=self.tf_log_dir)
        unit.fit(X, n_epochs=self.n_epochs, model_path=model_path, batch_size=self.batch_size,
                 checkpoint_steps=self.checkpoint_steps,
                 seed=self.seed)
        [reconstruction_loss] = unit.restore_and_eval(X, model_path=model_path, varlist=["reconstruction_loss"])
        if search_logging:
            with open(self.search_log_file, "a") as f:
                f.write("\t>> Reconstruction loss: {}...\n".format(reconstruction_loss))   
        if accepted_reconstruction_loss is None or reconstruction_loss <= accepted_reconstruction_loss:
            if search_logging:
                with open(self.search_log_file, "a") as f:
                    f.write("\t>> Accepted reconstruction loss\n")
                    f.write("\t>> Searching for better unit with the same n_neurons {} and reconstruction loss <= {}...\n".format(n_neurons, reconstruction_loss))

            # Recursively tries to find another unit with at least the "reconstruction_loss" but using stricter
            # L2 regularization (which is assumed to be better)
            better, better_model_path, better_reconstruction_loss = self.search_ordinary_unit_helper(X, n_neurons, l2_reg_range=(reg + self.epsilon, max_reg),
                                                                                                     accepted_reconstruction_loss=reconstruction_loss,
                                                                                                     n_trials=n_trials+1, search_logging=search_logging)
            if better is not None:
                assert(better_reconstruction_loss <= reconstruction_loss), "Invalid reconstruction loss"
                if search_logging:
                    with open(self.search_log_file, "a") as f:
                        f.write("\t>> Better unit found\n")
                return better, better_model_path, better_reconstruction_loss
            else:
                if search_logging:
                    with open(self.search_log_file, "a") as f:
                        f.write("\t>> Better unit not found\n")
                return unit, model_path, reconstruction_loss
        else:
            return None, None, None

    def search_ordinary_unit(self, X, n_neurons, l2_reg_range=(0.001,0.01), search_logging = True):
        min_reg, max_reg = l2_reg_range
        if search_logging:
            with open(self.search_log_file, "a") as f:
                f.write("\nSearching for ordinary unit of {} neurons, regularizer ({}, {})\n".format(n_neurons, min_reg, max_reg))
        return self.search_ordinary_unit_helper(X, n_neurons, l2_reg_range, n_trials=0, search_logging=search_logging)

    def search_gaussian_denoise_unit_helper(self, X, n_neurons, noise_stddev_range=(0.1,0.5), accepted_reconstruction_loss = None, n_trials = 0, search_logging = True):
        min_noise_stddev, max_noise_stddev = noise_stddev_range
        if n_trials > self.max_search_trials or min_noise_stddev > max_noise_stddev:
            return None, None, None
        n_inputs = X.shape[1]
        noise_stddev = (min_noise_stddev + max_noise_stddev) / 2
        if search_logging:
            with open(self.search_log_file, "a") as f:
                f.write("\t>> Constructing and training unit of {} neurons, noise_stddev {}, trial {}...\n".format(n_neurons, noise_stddev, n_trials))
        name = unit_config_str("denoise", n_inputs, n_neurons, hidden_activation=self.hidden_activation, noise_stddev=noise_stddev)
        model_path = os.path.join(self.cache_dir, name, "{}.model".format(name))
        unit = UnitAutoencoder(name=name,
                               n_inputs=n_inputs,
                               n_neurons=n_neurons,
                               hidden_activation=self.hidden_activation,
                               noise_stddev=noise_stddev,
                               tf_log_dir=self.tf_log_dir)
        unit.fit(X, n_epochs=self.n_epochs, model_path=model_path, batch_size=self.batch_size,
                 checkpoint_steps=self.checkpoint_steps,
                 seed=self.seed)
        [reconstruction_loss] = unit.restore_and_eval(X, model_path=model_path, varlist=["reconstruction_loss"])
        if search_logging:
            with open(self.search_log_file, "a") as f:
                f.write("\t>> Reconstruction loss: {}...\n".format(reconstruction_loss))            
        if accepted_reconstruction_loss is None or reconstruction_loss <= accepted_reconstruction_loss:
            if search_logging:
                with open(self.search_log_file, "a") as f:
                    f.write("\t>> Accepted reconstruction loss\n")
                    f.write("\t>> Searching for better unit with the same n_neurons {} and reconstruction loss <= {}...\n".format(n_neurons, reconstruction_loss))

            # Recursively tries to find another unit with at least the "reconstruction_loss" but with higher level of noises to the input                    
            better, better_model_path, better_reconstruction_loss = self.search_gaussian_denoise_unit_helper(X, n_neurons,
                                                                                                             noise_stddev_range=(noise_stddev + self.epsilon, max_noise_stddev),
                                                                                                             accepted_reconstruction_loss=reconstruction_loss,
                                                                                                             n_trials=n_trials+1, search_logging=False)
            if better is not None:
                assert(better_reconstruction_loss <= reconstruction_loss), "Invalid reconstruction loss"
                if search_logging:
                    with open(self.search_log_file, "a") as f:
                        f.write("\t>> Better unit found\n")
                return better, better_model_path, better_reconstruction_loss
            else:
                if search_logging:
                    with open(self.search_log_file, "a") as f:
                        f.write("\t>> Better unit not found\n")                
                return unit, model_path, reconstruction_loss
        else:
            return None, None
    
    def search_gaussian_denoise_unit(self, X, n_neurons, noise_stddev_range=(0.1,0.5), search_logging = True):
        min_noise_stddev, max_noise_stddev = noise_stddev_range
        if search_logging:
            with open(self.search_log_file, "a") as f:
                f.write("\n{}\n".format(timestr()))
                f.write("Searching for Gaussian denoise unit of {} neurons, noise_stddev ({}, {})\n".format(n_neurons, min_noise_stddev, max_noise_stddev))
        return self.search_gaussian_denoise_unit_helper(X, n_neurons, noise_stddev_range, n_trials=0, search_logging=search_logging)

    def search_dropout_unit_helper(self, X, n_neurons, dropout_rate_range=(0.1,0.5), n_trials = 0, search_logging = True):
        min_dropout_rate, max_dropout_rate = dropout_rate_range
        if n_trials > self.max_search_trials or min_dropout_rate > max_dropout_rate:
            return None, None
        n_inputs = X.shape[1]
        dropout_rate = (min_dropout_rate + max_dropout_rate) / 2
        if search_logging:
            with open(self.search_log_file, "a") as f:
                f.write("\t>> Training unit of {} neurons, dropout_rate {}, trial {}...\n".format(n_neurons, dropout_rate, n_trials))
        name = unit_config_str("dropout", n_inputs, n_neurons, hidden_activation=self.hidden_activation, dropout_rate=dropout_rate)
        model_path = os.path.join(self.cache_dir, name, "{}.model".format(name))
        unit = UnitAutoencoder(name=name,
                               n_inputs=n_inputs,
                               n_neurons=n_neurons,
                               hidden_activation=self.hidden_activation,
                               dropout_rate=dropout_rate,
                               tf_log_dir=self.tf_log_dir)
        unit.fit(X, n_epochs=self.n_epochs, model_path=model_path, batch_size=self.batch_size,
                 checkpoint_steps=self.checkpoint_steps,
                 seed=self.seed)
        [reconstruction_loss] = unit.restore_and_eval(X, model_path=model_path, varlist=["reconstruction_loss"])
        if search_logging:
            with open(self.search_log_file, "a") as f:
                f.write("\t>> Reconstruction loss: {}...\n".format(reconstruction_loss))            
        if reconstruction_loss <= self.accepted_reconstruction_loss:
            if search_logging:
                with open(self.search_log_file, "a") as f:
                    f.write("\t>> Accepted reconstruction loss\n")
                    f.write("\t>> Searching for better unit...\n")
            better, better_model_path = self.search_dropout_unit_helper(X, n_neurons,
                                                                        dropout_rate_range=(dropout_rate + self.epsilon, max_dropout_rate),
                                                                        n_trials=n_trials+1, search_logging=False)
            if better is not None:
                if search_logging:
                    with open(self.search_log_file, "a") as f:
                        f.write("\t>> Better unit found\n")
                return better, better_model_path
            else:
                if search_logging:
                    with open(self.search_log_file, "a") as f:
                        f.write("\t>> Better unit not found\n")                
                return unit, model_path
        else:
            return None, None
    
    def search_dropout_unit(self, X, n_neurons, dropout_rate_range=(0.1,0.5), search_logging = True):
        min_dropout_rate, max_dropout_rate = dropout_rate_range
        if search_logging:
            with open(self.search_log_file, "a") as f:
                f.write("\n{}\n".format(timestr()))
                f.write("Searching for dropout denoise unit of {} neurons, dropout rate ({}, {})\n".format(n_neurons, min_dropout_rate, max_dropout_rate))
        return self.search_dropout_unit_helper(X, n_neurons, dropout_rate_range, n_trials=0, search_logging=search_logging)
    
    def search_unit_helper(self, X, n_neurons_range, accepted_reconstruction_loss, n_trials = 0):
        """
        Search for a unit to reconstruct a given input with the number of hidden
        neurons within a given range

        Params:

        Return: 
        """
        min_neurons, max_neurons = n_neurons_range
        if n_trials > self.max_search_trials or min_neurons > max_neurons:
            return None, None
        n_inputs = X.shape[1]
        n_neurons = int((min_neurons + max_neurons)/2)
        if self.unit_type == "ordinary":
            unit, model_path, reconstructed_loss = self.search_ordinary_unit(X, n_neurons, l2_reg_range=self.l2_reg_range,
                                                                             accepted_reconstruction_loss=accepted_reconstruction_loss)
        elif self.unit_type == "denoise":
            unit, model_path, reconstructed_loss = self.search_gaussian_denoise_unit(X, n_neurons, noise_stddev_range=self.noise_stddev_range,
                                                                                     accepted_reconstruction_loss=accepted_reconstruction_loss)
        elif self.unit_type == "dropout":
            unit, model_path, reconstructed_loss = self.search_dropout_unit(X, n_neurons, dropout_rate_range=self.dropout_rate_range,
                                                                            accepted_reconstruction_loss=accepted_reconstruction_loss)
        else:
            raise ValueError("Invalid unit type {}".format(unit_type))
        assert(reconstruction_loss <= accepted_reconstruction_loss), "Invalid reconstruction loss"
        if unit is not None:
            left, left_model_path = self.search_unit_helper(X, n_neurons_range=(min_neurons, n_neurons - 1), n_trials=n_trials+1)
            if left is not None:
                return left, left_model_path
            else:
                return unit, model_path
        else:
            return self.search_unit_helper(X, n_neurons_range=(n_neurons + 1, max_neurons), n_trials=n_trials+1)

    def search_unit(self, X, n_neurons_range, accepted_reconstruction_loss):
        return self.search_unit_helper(X, n_neurons_range, accepted_reconstruction_loss)
        
    def doit(self, model_path = None, search_logging = True):
        self.autoencoder = StackedAutoencoders(name=self.name, cache_dir=self.cache_dir, tf_log_dir=self.tf_log_dir)
        if search_logging:
            with open(self.search_log_file, "a") as f:
                f.write("\n{}\n".format(timestr()))
        input_tensor = self.X
        for idx in range(self.n_units):
            if search_logging:
                with open(self.search_log_file, "a") as f:
                    f.write("\nSEARCH FOR UNIT AT LAYER {}\n".format(idx))
            n_inputs = input_tensor.shape[1]
            n_neurons_range = self._get_neurons_range(n_inputs)
            unit, unit_model_path = self.search_unit(input_tensor, n_neurons_range)
            if unit is not None:
                self.autoencoder.stack_autoencoder(unit=unit, layer_name="layer{}".format(idx))
                [input_tensor] = unit.restore_and_eval(input_tensor, model_path=unit_model_path, varlist=["hidden_outputs"])
                if search_logging:
                    with open(self.search_log_file, "a") as f:
                        f.write("UNIT AT LAYER {} FOUND: {}\n".format(idx, unit_model_path))
            else:
                self.autoencoder = None
                if search_logging:
                    with open(self.search_log_file, "a") as f:
                        f.write("UNIT AT LAYER {} NOT FOUND\n".format(idx))
                break
        if self.autoencoder is not None and model_path is not None:
            self.autoencoder.finalize()
            if search_logging:
                with open(self.search_log_file, "a") as f:
                    f.write("\nSAVING STACKED AUTOENCODER TO {}\n".format(model_path))
            self.autoencoder.save(model_path)
        return self.autoencoder
        
