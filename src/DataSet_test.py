import numpy as np
import unittest
import DataSet as myds

class TestDataSet(unittest.TestCase):

    def setUp(self):
        self.data_set = myds.DataSet("../input")
    
    def test_read_channel_file(self):
        channel = self.data_set.read_channel_file("Z", "Z001.txt")
        self.assertEqual(len(channel), 4097)
        self.assertTrue(np.array_equal(channel[0:3], [12, 22, 35]))
        self.assertTrue(np.array_equal(channel[-3:], [-11, 8, 77]))

    def test_read_channels_in_set(self):
        channels = self.data_set.read_channels_in_set("B")
        self.assertEqual(len(channels.keys()), 100)
        channel = channels["O001.txt"]
        self.assertTrue(np.array_equal(channel[:3], [-24, -22, -17]))
        self.assertTrue(np.array_equal(channel[-3:], [22, 9, -74]))

    def test_read_channels_all_sets(self):
        channels = self.data_set.read_channels_all_sets()
        self.assertEqual(len(channels.keys()), 5)
        channel = channels["C"]["N001.TXT"]
        self.assertTrue(np.array_equal(channel[:3], [-42, -39, -35]))
        self.assertTrue(np.array_equal(channel[-3:], [-37, -40, -64]))

        channel = channels["D"]["F001.txt"]
        self.assertTrue(np.array_equal(channel[:3], [34, 33, 28]))
        self.assertTrue(np.array_equal(channel[-3:], [39, 41, 7]))

        channel = channels["E"]["S001.txt"]
        self.assertTrue(np.array_equal(channel[:3], [100, 124, 153]))
        self.assertTrue(np.array_equal(channel[-3:], [57, 95, 462]))

    def test_create_segment_data(self):
        keys_count = self.data_set.segment_data_df.shape[0]
        features_count = self.data_set.segment_data_df.shape[1]
        self.assertEqual(keys_count, 11500)
        self.assertEqual(features_count, 179)
        v = self.data_set.segment_data_df.loc["C_N100.TXT_0"]
        self.assertEqual(len(v), 179)
        self.assertTrue(np.array_equal(v[:3], [-62, -64, -62]))
        
if __name__ == '__main__':
    unittest.main()
