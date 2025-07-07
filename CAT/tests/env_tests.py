import context
import unittest
import numpy as np
from src.environment import ToolFetchingEnvironment

class ToolFetchingEnvironmentTest(unittest.TestCase):
    def test_init(self):
        fetcher_pos = np.array([0, 0])
        worker_pos = np.array([5, 5])
        stn_pos = [np.array([2,2]), np.array([4,4])]
        tool_pos = [np.array([3,3]), np.array([6,6])]
        width = 10
        height = 10
        worker_goal = 0
        env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stn_pos, tool_pos, worker_goal, width, height)
        obs = env.reset()
        obs_w = obs[0]
        obs_f = obs[1]
        curr_w_pos, curr_f_pos, s_pos, curr_t_pos, f_tool, w_action, f_action, w_goal = obs_w
        self.assertTrue(np.array_equal(curr_w_pos, np.array([5,5])))
        self.assertTrue(np.array_equal(curr_f_pos, np.array([0, 0])))
        self.assertTrue(np.array_equal(s_pos[0], np.array([2, 2])))
        self.assertTrue(np.array_equal(s_pos[1], np.array([4, 4])))
        self.assertTrue(np.array_equal(curr_t_pos[0], np.array([3, 3])))
        self.assertTrue(np.array_equal(curr_t_pos[1], np.array([6, 6])))
        self.assertEqual(w_action, ToolFetchingEnvironment.WORKER_ACTIONS.NOOP)
        self.assertEqual(f_action, ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP)
        self.assertEqual(w_goal, 0)

    def test_move(self):
        fetcher_pos = np.array([0, 0])
        worker_pos = np.array([5, 5])
        stn_pos = [np.array([2,2]), np.array([4,4])]
        tool_pos = [np.array([3,3]), np.array([6,6])]
        width = 10
        height = 10
        worker_goal = 0
        env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stn_pos, tool_pos, worker_goal, width, height)
        env.reset()
        obs, reward, done, info = env.step([ToolFetchingEnvironment.WORKER_ACTIONS.LEFT, (ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT, None)])
        obs_w = obs[0]
        obs_f = obs[1]
        curr_w_pos, curr_f_pos, s_pos, curr_t_pos, f_tool, w_action, f_action, w_goal = obs_w
        self.assertTrue(np.array_equal(curr_w_pos, np.array([4,5])))
        self.assertTrue(np.array_equal(curr_f_pos, np.array([0, 0])))
        self.assertTrue(np.array_equal(s_pos[0], np.array([2, 2])))
        self.assertTrue(np.array_equal(s_pos[1], np.array([4, 4])))
        self.assertTrue(np.array_equal(curr_t_pos[0], np.array([3, 3])))
        self.assertTrue(np.array_equal(curr_t_pos[1], np.array([6, 6])))
        self.assertEqual(w_action, ToolFetchingEnvironment.WORKER_ACTIONS.LEFT)
        self.assertEqual(f_action, ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT)
        self.assertEqual(w_goal, 0)





if __name__ == '__main__':
    unittest.main()
