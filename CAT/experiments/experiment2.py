"""
Runs an experiment where fetcher asks about different goals
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
    results['graph 1'] = {}
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
    results['graph 1']['X'] = []
    for t in range(3):
        results['graph 1'][t] = []
    for g in range(3):
        paths = opt_path_perm(stn_pos[g], worker_pos)
        env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stn_pos, tool_pos, g)
        for path in paths:
            obs = env.reset()
            done = [False, False]
            fetcher = FetcherQueryPolicy()
            worker = PlanPolicy(path + (ToolFetchingEnvironment.WORKER_ACTIONS.WORK,))
            cost = 0
            while not done[0]:
                if args.render:
                    env.render()
                    sleep(0.05)
                obs, reward, done, _ = env.step([worker(obs[0]), fetcher(obs[1])])
                cost += reward[1]
            results['graph 1']['X'].append(int(cost))
        for t in range(3):
            time = 0
            def queryStrat(obs, agent):
                w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
                if np.max(agent.probs) < 1 and \
                np.array_equal(f_pos, t_pos[0]) and \
                agent.probs[t] > 0:
                    return [t]
                return None
            for path in paths:
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
                results['graph 1'][t].append(int(cost))


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


