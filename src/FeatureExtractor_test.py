import os
import numpy as np
import unittest
from DataSet import DataSet
from FeatureExtractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.data_set = DataSet("../input")
        self.data_set.load_train(os.path.join(self.data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TRAIN.csv"))
        self.data_set.load_test(os.path.join(self.data_set.cache_dir, "segment_numseg23_target@AB@CD@E@_ratio0.2_rand0_TEST.csv"))

    def test_identity(self):
        feature_extractor = FeatureExtractor(self.data_set.train_df, scaler = None)
        feature_extractor.fit()
        X_coded, y = feature_extractor.eval()
        self.assertEqual(X_coded.shape[0], self.data_set.train_df.shape[0])
        self.assertEqual(X_coded.shape[1] + 1, self.data_set.train_df.shape[1])
        self.assertTrue(np.array_equal(X_coded, self.data_set.train_df.iloc[:,:178]))
        self.assertTrue(np.array_equal(y, self.data_set.train_df.iloc[:,178]))
        
if __name__ == '__main__':
    unittest.main()
