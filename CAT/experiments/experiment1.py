"""
Runs a basic experiment where fetcher queries at specific timestep
"""
import context
from src.environment import ToolFetchingEnvironment
import numpy as np
from src.agents.agent import RandomWorkerPolicy
from src.agents.agent import PlanPolicy
from src.agents.agent_adhoc_q import FetcherQueryPolicy
from src.agents.agent_adhoc_q import FetcherAltPolicy
from itertools import permutations
import pandas as pd
import json
import argparse
from time import sleep


def experiment(args):
    fetcher_pos = np.array([4, 0])
    worker_pos = np.array([5, 0])
    stn_pos = [np.array([7,2]), np.array([7,8]), np.array([3, 8])]
    tool_pos = [np.array([2,3]) for _ in range(3)]
    results = {}
    for g in range(3):
        results[f'goal {g}'] = {}

        def opt_path_perm(stn_pos, worker_pos):
            worker_path = []
            offset = stn_pos-worker_pos

            if offset[0] >= 0:
                worker_path += [ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT] * offset[0]
            else:
                worker_path += [ToolFetchingEnvironment.WORKER_ACTIONS.LEFT] * -offset[0]

            if offset[1] >= 0:
                worker_path += [ToolFetchingEnvironment.WORKER_ACTIONS.UP] * offset[1]
            else:
                worker_path += [ToolFetchingEnvironment.WORKER_ACTIONS.DOWN] * -offset[1]

            return list(set(permutations(worker_path)))

        paths = opt_path_perm(stn_pos[g], worker_pos)
        env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stn_pos, tool_pos, g)
        results[f'goal {g}']['X'] = []
        for path in paths:
            obs = env.reset()
            done = [False, False]
            fetcher = FetcherQueryPolicy()
            worker = PlanPolicy(path + (ToolFetchingEnvironment.WORKER_ACTIONS.WORK,))
            cost = 0
            while not done[0]:
                obs, reward, done, _ = env.step([worker(obs[0]), fetcher(obs[1])])
                cost += reward[1]
            results[f'goal {g}']['X'].append(int(cost))
        for t in range(len(paths[0])):
            results[f'goal {g}'][str(t)] = []
            time = 0
            def queryStrat(obs, agent):
                if np.max(agent.probs) == 1:
                    return None
                if time >= t:
                    return [g]
                return None
            for path in paths:
                time = 0
                obs = env.reset()
                done = [False, False]
                fetcher = FetcherQueryPolicy(queryStrat)
                worker = PlanPolicy(path + (ToolFetchingEnvironment.WORKER_ACTIONS.WORK,))
                cost = 0
                while not done[0]:
                    if args.render:
                        env.render()
                        sleep(0.05)
                    obs, reward, done, _ = env.step([worker(obs[0]), fetcher(obs[1])])
                    cost += reward[1]
                    time += 1
                    #input()
                results[f'goal {g}'][str(t)].append(int(cost))
        env.close()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='results.json', help='Output File')
    parser.add_argument('--render', action='store_true', help='Flag should be present if rendering is desired')

    args = parser.parse_args()

    results = experiment(args)
    with open(args.output, 'w') as f:
        json.dump(results, f)


