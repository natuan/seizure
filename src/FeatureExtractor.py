import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(self, X_train, y_train, scaler = MinMaxScaler(), extractor = None, model_path = None):
        """
        Constructor
        
        Params:
        - scaler: a scaler for preprocessing the features
        - extractor: if None then scaled inputs are features; otherwise an instance of Autoencoder
        - model_path: the model full path file to be restored; ignored if None

        Return: None
        """
        self.scaler = scaler
        self.X_scaled = self.scaler.fit_transform(X_train) if (self.scaler) else X_train
        self.y_train = y_train
        self.extractor = extractor
        self.model_path = model_path
        
    def fit(self, model_path, n_epochs, batch_size = 256, seed = 42, tfdebug = False):
        """
        Train the feature extraction algorithm against the features self.X_scaled based on the choice
        of self.extractor

        Params: 
        - n_epochs, batch_size, seed
        - model_path: model file in full path to be saved; ignored if None

        Return: None
        """
        assert(model_path is not None), "Invalid model path"
        if (self.extractor == None):
            pass
        else:
            self.model_path = model_path
            self.extractor.fit(X_train=self.X_scaled, n_epochs=n_epochs, batch_size=batch_size, seed=seed, model_path=self.model_path, tfdebug=tfdebug)
    
    def codings(self,
                X = None,
                inverse_transform = False):
        """
        Apply the trained extractor to a pandas data frame to retrieve the coding of the feature part

        Params:
        - X: same as self.X_train

        Return: X_coded, the coding representations of the features X
        """
        assert(self.model_path is not None), "Invalid model path"
        if X is not None:
            X_scaled = self.scaler.transform(X) if (self.scaler) else X
        else:
            X_scaled = self.X_scaled
        if self.extractor:
            [X_coded] = self.extractor.restore_and_eval(X_scaled, self.model_path, ["hidden_outputs"])
        else:
            X_coded = X_scaled
        if inverse_transform:
            assert(self.scaler), "Invalid scaler for inverse transform"
            self.scaler.inverse_transform(X_coded)
        return X_coded

    def outputs(self,
                X = None,
                inverse_transform = False,
                sample_indices_to_plot = [],
                plot_dir_path = None,
                tfdebug = False):
        """
        Evaluate the result of the output layer given an input sample set. Optionally to plot and save
        into file some samples and the corresponding recovered outputs
        """
        assert(self.model_path is not None), "Invalid model path"
        if X is not None:
            X_scaled = self.scaler.transform(X) if (self.scaler) else X
        else:
            X_scaled = self.X_scaled
        if self.extractor:
            [X_outputs] = self.extractor.restore_and_eval(X_scaled, self.model_path, ["outputs"], tfdebug=tfdebug)
        else:
            X_outputs = X_scaled
        if inverse_transform:
            assert(self.scaler), "Invalid scaler for inverse transform"
            X_outputs = self.scaler.inverse_transform(X_outputs)           
        if plot_dir_path is not None and not os.path.exists(plot_dir_path):
            os.makedirs(plot_dir_path)
        for idx_to_plot in sample_indices_to_plot:
            fig, axes = plt.subplots(2, 1, figsize=(15, 8))
            axes[0].plot(X[idx_to_plot], label="X[{}]".format(idx_to_plot))
            axes[0].set_xlabel("X[{}]".format(idx_to_plot))
            axes[0].set_ylabel("Signal value")
            axes[1].plot(X_outputs[idx_to_plot], label="X_outputs[{}]".format(idx_to_plot))
            axes[1].set_xlabel("X_outputs[{}]".format(idx_to_plot))
            axes[1].set_ylabel("Signal value")
            plot_file = os.path.join(plot_dir_path, "X_vs_Xoutputs_{}.png".format(idx_to_plot))
            plt.savefig(plot_file)
        return X_outputs
