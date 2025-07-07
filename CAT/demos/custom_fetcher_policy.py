"""
Demo that shows how to write a custom policy
This new policy has fetcher wait until worker has arrived at a station and then fetch the tool
"""
import context
from src.environment import ToolFetchingEnvironment
import numpy as np
from src.agents.agent import RandomWorkerPolicy
from src.agents.agent import Policy
from time import sleep

class CustomFetcherPolicy(Policy):
    """
    Custom Fetcher Policy
    Waits until worker arrives at toolbox
    """

    def action_to_goal(self, pos, goal):
        actions = []
        if pos[0] < goal[0]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT)
        elif pos[0] > goal[0]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT)
        if pos[1] > goal[1]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN)
        elif pos[1] < goal[1]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.UP)
        if len(actions) == 0:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP
        return np.random.choice(actions)


    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if w_action != ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP, None
        for i,pos in enumerate(s_pos):
            if np.array_equal(pos, w_pos):
                target = i
                break

        if f_tool != target:
            if np.array_equal(f_pos, t_pos[target]):
                return ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP, target
            return self.action_to_goal(f_pos, t_pos[target]), None
        else:
            return self.action_to_goal(f_pos, s_pos[target]), None


if __name__ == '__main__':
    #Fetcher start position
    fetcher_pos = np.array([4, 0])
    #Worker Start Position
    worker_pos = np.array([5, 0])
    #List of Station Positions
    stn_pos = [np.array([7,2]), np.array([7,8]), np.array([3, 8])]
    #List of tool positions
    tool_pos = [np.array([2,3]), np.array([3,3]), np.array([7,4])]
    env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stn_pos, tool_pos, 2)
    #grab observation
    obs = env.reset()
    done = [False, False]
    #Need Policy to handle multiple tool positions
    fetcher = CustomFetcherPolicy()
    worker = RandomWorkerPolicy()
    #run until done
    while not done[0]:
        #only needed for rendering
        env.render()
        sleep(0.05)

        obs, reward, done, _ = env.step([worker(obs[0]), fetcher(obs[1])])
    env.close()
