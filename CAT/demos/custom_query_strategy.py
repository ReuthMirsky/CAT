"""
Shows how to write a custom query strategy with existing fetcher polices
"""
import context
from src.environment import ToolFetchingEnvironment
import numpy as np
from src.agents.agent import RandomWorkerPolicy
from src.agents.agent_adhoc_q import FetcherAltPolicy
from time import sleep

#query function takes the observation and the agent as arguments, can access agent's current belief state
def query_at_begining(obs, agent):
    if np.max(agent.probs < 1):
        valid = [i for i,p in enumerate(agent.probs) if p > 0]
        #return random sample of half valid stations
        return np.random.choice(valid, size=len(valid)//2, replace=False)
    return None

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
    #Set query policy flag to custom query function
    fetcher = FetcherAltPolicy(query_policy=query_at_begining)
    worker = RandomWorkerPolicy()
    #run until done
    while not done[0]:
        #only needed for rendering
        env.render()
        sleep(0.05)

        obs, reward, done, _ = env.step([worker(obs[0]), fetcher(obs[1])])
    env.close()
