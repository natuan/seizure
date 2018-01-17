import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
        
    def fit(self, model_path, n_epochs, batch_size = 256, seed = 42):
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
            self.extractor.fit(self.X_scaled, n_epochs=n_epochs, batch_size=batch_size, seed=seed, model_path = self.model_path)
    
    def codings(self, X = None, inverse_transform = False):
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
            results = self.extractor.restore_and_eval(X_scaled, self.model_path, ["hidden_outputs"])
            X_coded = results["hidden_outputs"]
        else:
            X_coded = X_scaled
        if inverse_transform:
            self.extractor.inverse_transform(X_coded)            
        return X_coded

    def outputs(self, X = None, inverse_transform = False):
        assert(self.model_path is not None), "Invalid model path"
        if X is not None:
            X_scaled = self.scaler.transform(X) if (self.scaler) else X
        else:
            X_scaled = self.X_scaled
        if self.extractor:
            results = self.extractor.restore_and_eval(X_scaled, self.model_path, ["outputs"])
            X_outputs = results["outputs"]
        else:
            X_outputs = X_scaled
        if inverse_transform:
            self.extractor.inverse_transform(X_outputs)
        return X_outputs
