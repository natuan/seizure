import os
import numpy as np
import unittest
from sklearn.preprocessing import MinMaxScaler
from DataSet import DataSet
from FeatureExtractor import FeatureExtractor
from Denoise import UnitAutoencoder

class TestDenoiseAutoencoder(unittest.TestCase):
    def setUp(self):
        self.data_set = DataSet("/home/natuan/MyHDD/ml_nano_capstone/input")
        self.X_train, self.y_train = self.data_set.load_features_and_target(os.path.join(self.data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TRAIN.csv"))

    def test_unit_autoencoder(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        extractor = UnitAutoencoder(name="unit", n_inputs = self.X_train.shape[1], n_neurons = 100, dropout_rate = 0.3)
        feature_extractor = FeatureExtractor(scaler=scaler, extractor=extractor)
        feature_extractor.fit(self.X_train, n_epochs = 5)
        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()

