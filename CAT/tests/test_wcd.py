import context
import unittest
import src.wcd_utils as wcd_utils
import src.acd_utils as acd_utils
import numpy as np

class WCDTEST(unittest.TestCase):

    def test_fastWcd_2goals(self):
        pos = np.array([5, 0])
        goals = [np.array([0, 5]), np.array([10, 5])]
        wcd = wcd_utils.fast_wcd(pos, goals)
        self.assertTrue(wcd == 5)

    def test_fastWcd_1goal(self):
        pos = np.array([5, 0])
        goals = [np.array([0, 5]), np.array([0, 5])]
        wcd = wcd_utils.fast_wcd(pos, goals)
        self.assertTrue(wcd == 10)


    def test_WCD(self):
        goals = [np.array([0, 5]), np.array([10, 5]), np.array([0, 5])]
        wcd = acd_utils.WCD(goals)
        self.assertEqual(wcd[0,1][5,0], 6.0)
        self.assertEqual(wcd[0,2][5,0],  11.0)



if __name__ == '__main__':
    unittest.main()
