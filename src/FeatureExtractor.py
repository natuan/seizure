import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureExtractor:
    def __init__(self, scaler = MinMaxScaler(), extractor = None):
        """
        Constructor
        
        Params:
        - scaler: a scaler for preprocessing the features
        - extractor: if None then scaled inputs are features; otherwise an instance of Autoencoder

        Return: None
        """
        self.scaler = scaler
        self.extractor = extractor

    def _separate_features_target(self, segment_df):
        """
        Separate features and target from a segment data frame
        
        Params:
        - segment_df: a pandas data frame containing both features and target; should be either
        the train_df or test_df of a DataSet instance
        
        Return: (X, y) where X and y are numpy ndarray with the same shape with those of self.X and self.y
        """
        assert(segment_df is not None), "Invalid segment data frame"
        X = pd.DataFrame.as_matrix(segment_df.drop(["target_class"], axis = 1))
        y = pd.DataFrame.as_matrix(segment_df["target_class"])
        return X, y
        
    def fit(self, X, n_epochs = 100, batch_size = 256, seed = 42):
        """
        Train the feature extraction algorithm against the features self.X based on the choice
        of self.extractor
        """
        X_scaled = self.scaler.fit_transform(X) if (self.scaler) else X
        if (self.extractor == None):
            pass
        else:
            self.extractor.fit(X_scaled, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    
    def eval(self, X):
        """
        Apply the trained extractor to a pandas data frame to retrieve the coding of the feature part

        Params:
        - X: features

        Return: X_coded, the coding representations of the features X
        """
        X_scaled = self.scaler.transform(X) if (self.scaler) else X
        X_coded = self.extractor.outputs(X_scaled) if self.extractor else X_scaled
        return X_coded
