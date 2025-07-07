"""
This file provides an example of how to create an environment and run it with provided policies
"""
#gives file access to other src directory
import context
from src.environment import ToolFetchingEnvironment
import numpy as np
from src.agents.agent import RandomWorkerPolicy
from src.agents.agent_adhoc_q import FetcherQueryPolicy
from time import sleep


if __name__ == '__main__':
    #Fetcher start position
    fetcher_pos = np.array([4, 0])
    #Worker Start Position
    worker_pos = np.array([5, 0])
    #List of Station Positions
    stn_pos = [np.array([7,2]), np.array([7,8]), np.array([3, 8])]
    #List of tool positions, in this example they are all located in the same spot
    tool_pos = [np.array([2,3]) for _ in range(3)]
    env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stn_pos, tool_pos, 2)
    #grab observation
    obs = env.reset()
    done = [False, False]
    fetcher = FetcherQueryPolicy()
    worker = RandomWorkerPolicy()
    #run until done
    while not done[0]:
        #only needed for rendering
        env.render()
        sleep(0.05)

        obs, reward, done, _ = env.step([worker(obs[0]), fetcher(obs[1])])
    env.close()
