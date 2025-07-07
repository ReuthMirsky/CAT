"""
Runs experimental analaysis of query timing strategies in a complicated domain
"""
import context
import random
from src.environment import ToolFetchingEnvironment
from src.agents.classifier import EpsilonGreedyClassifier
import numpy as np
from src.agents.agent import RandomWorkerPolicy
from src.agents.agent import PlanPolicy, SubOptimalWorker
from src.agents.agent_adhoc_q import FetcherQueryPolicy
from src.agents.agent_adhoc_q import FetcherAltPolicy, FetcherAgentTypePolicy
from src.agents.agent_adhoc_q import never_query
from itertools import permutations
import pandas as pd
import json
import argparse
from time import sleep






def experiment(args):
    global time
    global rand_time
    results = {}
    results['graph 1'] = {}
    epsilons = list(args.epsilons)
    epsilons.sort()
    median = epsilons[len(epsilons)//2]
    strats = {'Min':EpsilonGreedyClassifier([epsilons[0]]), 'Median': EpsilonGreedyClassifier([median]), 'Max':EpsilonGreedyClassifier([epsilons[-1]]), 'Joint Prob Distribution':EpsilonGreedyClassifier(epsilons)}
    def dist(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    for strat in strats:
        results['graph 1'][strat] = []
    results['graph 1']['baseline'] = []
    for t in range(args.num_exp):
        print(f"exp num: {t}")
        if args.cluster_stations:
            num_clusters = args.num_stations//4
            cluster_pos = [(i,j) for i in range(args.grid_size//2) for j in range(args.grid_size//2)]
            cluster_pos = random.sample(cluster_pos, num_clusters)
            stations_pos = np.array([k for i,j in cluster_pos for k in [(2*i,2*j), (2*i+1, 2*j), (2*i, 2*j+1), (2*i+1, 2*j+1)]])
        else:
            stations_pos = [(i,j) for i in range(args.grid_size) for j in range(args.grid_size)]
            stations_pos = np.array(random.sample(stations_pos, args.num_stations))
        pos = [(i,j) for i in range(args.grid_size) for j in range(args.grid_size)]
        tools_pos = np.array(random.sample(pos, args.num_stations))
        fetcher_pos = np.array(random.choice(pos))
        worker_pos = np.array(random.choice(pos))
        goal = int(random.random()*len(stations_pos))
        env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stations_pos, tools_pos, goal, width=args.grid_size, height=args.grid_size)
        for e in epsilons:
            results['graph 1']['baseline'].append(-int(max(dist(worker_pos, stations_pos[goal]), dist(fetcher_pos, tools_pos[goal])+dist(tools_pos[goal], stations_pos[goal]))))
            for strat in strats:
                print(f"strat: {strat}")
                obs = env.reset()
                done = [False, False]
                strats[strat].reset()
                fetcher = FetcherAgentTypePolicy(strats[strat])
                worker = SubOptimalWorker(e)
                cost = 0
                time = 0
                rand_time = int(random.random()*args.grid_size)
                while not done[0]:
                    if args.render:
                        env.render()
                        sleep(0.05)
                    obs, reward, done, _ = env.step([worker(obs[0]), fetcher(obs[1])])
                    cost += reward[1]
                    time += 1
                results['graph 1'][strat].append(int(cost))
                env.close()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='results.json', help='Output File')
    parser.add_argument('--render', action='store_true', help='Flag should be present if rendering is desired')
    parser.add_argument('--num_exp', default=100, type=int, help='Number of experiments to run')
    parser.add_argument('--grid_size', default=100, type=int, help='Grid Size of environment')
    parser.add_argument('--num_stations', default=400, type=int, help='Number of stations in environment')
    parser.add_argument('--cluster_stations', action='store_true', help='Flag should be present if stations are clustered')
    parser.add_argument('--epsilons', nargs='+', default=[0.01, 0.05, 0.1, 0.2], type=float, help='Epsilon Values considered in experiment')

    args = parser.parse_args()

    results = experiment(args)
    with open(args.output, 'w') as f:
        json.dump(results, f)


