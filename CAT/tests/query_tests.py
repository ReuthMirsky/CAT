import context
import unittest
import src.agents.query_policies as qp 
import src.wcd_utils as wcd_utils
import src.acd_utils as acd_utils
import numpy as np


class UtilsTest(unittest.TestCase):
    def test_is_ZB(self):
        w_pos = np.array([0, 0])
        f_pos = np.array([10, 10])
        s_pos = [np.array([5,5]), np.array([7, 7])]
        t_pos = [np.array([5, 5]), np.array([7, 7])]
        f_tool = None
        w_action = None
        f_action = None
        answer = None
        answer1 = qp.is_ZB((w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer), 0, 1)
        self.assertTrue(not answer1)
        f_pos = np.array([6,6])
        answer2 = qp.is_ZB((w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer), 0, 1)
        self.assertTrue(answer2)
        f_pos = np.array([5,7])
        answer3 = qp.is_ZB((w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer), 0, 1)
        self.assertTrue(answer3)

    def test_get_voi_test(self):
        goals = [np.array([0, 5]), np.array([10, 5])]
        edp = {}
        edp[(0,1)] = np.zeros((10,10))
        edp[(1,0)] = np.zeros((10,10))
        edp[(0,1)][5,0] = 10
        edp[(1,0)][5,0] = 10
        wcd = acd_utils.WCD(goals)
        self.assertEqual(wcd[0,1][5,0], 6.0)
        voi = qp.get_voi(0, [0,1], np.array([5, 0]), np.array([5,0]), wcd, edp)
        self.assertEqual(voi, 5)

    def test_get_voi_test2(self):
        goals = [np.array([0, 5]), np.array([10, 5]), np.array([0, 5])]
        edp = {}
        edp[(0,1)] = np.zeros((10,10))
        edp[(1,0)] = np.zeros((10,10))
        edp[(0,2)] = np.zeros((10,10))
        edp[(2,0)] = np.zeros((10,10))
        edp[(0,1)][5,0] = 13
        edp[(1,0)][5,0] = 13
        edp[(0,2)][5,0] = 15
        edp[(2,0)][5,0] = 15
        wcd = acd_utils.WCD(goals)
        self.assertEqual(wcd[0,1][5,0], 6.0)
        self.assertEqual(wcd[0,2][5,0],  11.0)
        voi = qp.get_voi(0, [0,1,2], np.array([5, 0]), np.array([5,0]), wcd, edp)
        self.assertEqual(voi, 10)


    def test_get_voi_test3(self):
        goals = [np.array([0, 5]), np.array([10, 5]), np.array([0, 5])]
        edp = {}
        edp[(0,1)] = np.zeros((10,10))
        edp[(1,0)] = np.zeros((10,10))
        edp[(0,2)] = np.zeros((10,10))
        edp[(2,0)] = np.zeros((10,10))
        edp[(0,1)][5,0] = 9
        edp[(1,0)][5,0] = 9
        edp[(0,2)][5,0] = 15
        edp[(2,0)][5,0] = 15
        wcd = acd_utils.WCD(goals)
        self.assertEqual(wcd[0,1][5,0], 6.0)
        self.assertEqual(wcd[0,2][5,0],  11.0)
        voi = qp.get_voi(0, [0,1,2], np.array([5, 0]), np.array([5,0]), wcd, edp)
        self.assertEqual(voi, 9)



if __name__ == '__main__':
    unittest.main()
