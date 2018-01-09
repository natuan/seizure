import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureExtractor:
    def __init__(self, segment_df, scaler = MinMaxScaler(), extractor = "raw"):
        """
        Constructor
        
        Params:
        - segment_df: data frame from which features will be extracted; this data frame
        contains also the target class and should be the train_df of a DataSet
        - scaler: a scaler for preprocessing the features
        - extractor: name of the feature extraction algorithm

        Return: None
        """
        self.X, self.y = self._separate_features_target(segment_df) # numpy ndarray, not data frames
        self.scaler = scaler
        self.extractor = extractor
        self.engine = None

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
        
    def fit(self):
        """
        Train the feature extraction algorithm against the features self.X based on the choice
        of self.extractor
        """
        assert(self.X is not None), "Invalid features"        
        X_scaled = self.scaler.fit_transform(self.X) if (self.scaler) else self.X
        if (self.extractor == "raw"):
            pass
        elif (self.extractor == "denoise"):
            pass
        else:
            assert(False), "Invalid feature extractor {}".format(self.extractor)
    
    def eval(self, segment_df = None):
        """
        Apply the trained extractor to a pandas data frame to retrieve the coding of the feature part

        Params:
        - segment_df: a pandas data frame containing both features and target parts. If None then self.X and self.y are used.

        Return: (X_coded, y) where X_coded is the coding representations of the feature part, and y is the
        target part of the segment data frame
        """
        assert(self.engine is not None or self.extractor == "raw"), "The extractor needs to be trained by calling fit(...) before being used for extracting features"
        if (segment_df is not None):
            X, y = self._separate_features_target(segment_df)
        else:
            X, y = self.X, self.y
        X_scaled = self.scaler.transform(X) if (self.scaler) else X
        X_coded = self.engine.eval(X_scaled, y) if self.engine else X_scaled
        return X_coded, y
