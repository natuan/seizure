import os
import numpy as np
import tensorflow as tf
import unittest
from sklearn.preprocessing import MinMaxScaler
from DataSet import DataSet
from FeatureExtractor import FeatureExtractor
from Denoise import UnitAutoencoder
from Utils import dicts_equal

class TestDenoiseAutoencoder(unittest.TestCase):
    initialized = False
    def setUp(self):
        if (not TestDenoiseAutoencoder.initialized):
            self.root_dir = "/home/natuan/MyHDD/ml_nano_capstone/"
            self.data_set = DataSet(input_dir=os.path.join(self.root_dir, "input"),
                                    cache_dir=os.path.join(self.root_dir, "cache"))
            self.X_train, self.y_train = self.data_set.load_features_and_target(os.path.join(self.data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TRAIN.csv"))
            TestDenoiseAutoencoder.initialized = True
    """
    def test_ordinary_autoencoder_0(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        unit_name = "ordinary_autoencoder_0"
        extractor = UnitAutoencoder(name=unit_name, n_inputs = self.X_train.shape[1], n_neurons = 179, root_logdir = self.tf_logdir)
        feature_extractor = FeatureExtractor(scaler=scaler, extractor=extractor)
        print("Train {} on {} segments...".format(unit_name, self.X_train.shape[0]))
        feature_extractor.fit(self.X_train, n_epochs = 2000)
        self.assertTrue(True)
        
    def test_ordinary_autoencoder_1(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        unit_name = "ordinary_autoencoder_1"
        extractor = UnitAutoencoder(name=unit_name, n_inputs = self.X_train.shape[1], n_neurons = 100, root_logdir = self.tf_logdir)
        feature_extractor = FeatureExtractor(scaler=scaler, extractor=extractor)
        print("Train {} on {} segments...".format(unit_name, self.X_train.shape[0]))
        feature_extractor.fit(self.X_train, n_epochs = 2000)
        self.assertTrue(True)

    def test_ordinary_autoencoder_2(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        unit_name = "ordinary_autoencoder_2"
        l2_reg = tf.contrib.layers.l2_regularizer(0.001)
        extractor = UnitAutoencoder(name=unit_name, n_inputs = self.X_train.shape[1], n_neurons = 100, root_logdir = self.tf_logdir, regularizer = l2_reg)
        feature_extractor = FeatureExtractor(scaler=scaler, extractor=extractor)
        print("Train {} on {} segments...".format(unit_name, self.X_train.shape[0]))
        feature_extractor.fit(self.X_train, n_epochs = 2000)
        self.assertTrue(True)       

    def test_ordinary_autoencoder_3(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        unit_name = "ordinary_autoencoder_3"
        extractor = UnitAutoencoder(name=unit_name, n_inputs = self.X_train.shape[1], n_neurons = 100, root_logdir = self.tf_logdir, tied_weights = True)
        feature_extractor = FeatureExtractor(scaler=scaler, extractor=extractor)
        print("Train {} on {} segments...".format(unit_name, self.X_train.shape[0]))
        feature_extractor.fit(self.X_train, n_epochs = 2000)
        self.assertTrue(True)       

    def test_gaussian_noise_autoencoder_0(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        unit_name = "gaussian_noise_autoencoder_0"
        extractor = UnitAutoencoder(name=unit_name, n_inputs = self.X_train.shape[1], n_neurons = 178, noise_stddev = 0.1, root_logdir = self.tf_logdir)
        feature_extractor = FeatureExtractor(scaler=scaler, extractor=extractor)
        print("Train {} on {} segments...".format(unit_name, self.X_train.shape[0]))
        feature_extractor.fit(self.X_train, n_epochs = 2000)
        self.assertTrue(True)             
    
    def test_gaussian_noise_autoencoder_1(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        unit_name = "gaussian_noise_autoencoder_1"
        extractor = UnitAutoencoder(name=unit_name, n_inputs = self.X_train.shape[1], n_neurons = 178, noise_stddev = 0.3, root_logdir = self.tf_logdir)
        feature_extractor = FeatureExtractor(scaler=scaler, extractor=extractor)
        print("Train {} on {} segments...".format(unit_name, self.X_train.shape[0]))
        feature_extractor.fit(self.X_train, n_epochs = 2000)
        self.assertTrue(True)             
    
    def test_gaussian_noise_autoencoder_2(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        unit_name = "gaussian_noise_autoencoder_2"
        extractor = UnitAutoencoder(name=unit_name, n_inputs = self.X_train.shape[1], n_neurons = 200, noise_stddev = 0.5, root_logdir = self.tf_logdir)
        feature_extractor = FeatureExtractor(scaler=scaler, extractor=extractor)
        print("Train {} on {} segments...".format(unit_name, self.X_train.shape[0]))
        feature_extractor.fit(self.X_train, n_epochs = 10000)
        self.assertTrue(True)             
    
    def test_gaussian_noise_autoencoder_3(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        unit_name = "gaussian_noise_autoencoder_3"
        batch_size = self.X_train.shape[0]
        extractor = UnitAutoencoder(name=unit_name, n_inputs = self.X_train.shape[1], n_neurons = 200, dropout_rate = 0.5, root_logdir = self.tf_logdir)
        feature_extractor = FeatureExtractor(scaler=scaler, extractor=extractor)
        print("Train {} on {} segments...".format(unit_name, self.X_train.shape[0]))
        feature_extractor.fit(self.X_train, n_epochs = 10000, batch_size=batch_size)
        self.assertTrue(True)             
    
    def test_gaussian_noise_autoencoder_4(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        unit_name = "gaussian_noise_autoencoder_4"
        batch_size = self.X_train.shape[0]
        tf_log_dir = "/home/natuan/MyHDD/ml_nano_capstone/test_gaussian_noise_autoencoder_4"
        extractor = UnitAutoencoder(name=unit_name, n_inputs = self.X_train.shape[1], n_neurons = 200, dropout_rate = 0.5, tf_log_dir = tf_log_dir)
        feature_extractor = FeatureExtractor(scaler=scaler, extractor=extractor)
        print("Train {} on {} segments...".format(unit_name, self.X_train.shape[0]))
        feature_extractor.fit(self.X_train, n_epochs = 500000, batch_size=batch_size)
        self.assertTrue(True)             
    """    
    def test_save_restore(self):
        n_inputs = self.X_train.shape[1]
        n_neurons = 10
        model_path = os.path.join(self.root_dir,"Denoise_test","model_final.ckpt")
        autoencoder_0 = UnitAutoencoder("MyAutoencoder", n_inputs, n_neurons)
        feature_extractor = FeatureExtractor(extractor=autoencoder_0) # (0,1) min-max scaler
        feature_extractor.fit(self.X_train, n_epochs = 10, model_path=model_path)

        autoencoder_1 = UnitAutoencoder("MyAutoencoder", n_inputs, n_neurons)
        autoencoder_1.restore(model_path)

        trained_params = autoencoder_0.params
        restored_params = autoencoder_1.params
        self.assertTrue(dicts_equal(trained_params, restored_params))
        
if __name__ == '__main__':
    unittest.main()

