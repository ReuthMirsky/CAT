"""
Runs experimental analaysis of query timing strategies in a complicated domain
"""
import context
import random
from src.environment import ToolFetchingEnvironment
import numpy as np
from src.agents.agent import RandomWorkerPolicy
from src.agents.agent import PlanPolicy
from src.agents.agent_adhoc_q import FetcherQueryPolicy
from src.agents.agent_adhoc_q import FetcherAltPolicy
from src.agents.agent_adhoc_q import never_query, random_query, max_action_query, min_action_query, median_action_query, smart_query
from itertools import permutations
import pandas as pd
import json
import argparse
from time import sleep

#strats = {'Never Query':never_query, 'Random Query':random_query, "Max Action Query":max_action_query, "Min Action Query":min_action_query, "Median Action Query":median_action_query, "Smart Query":smart_query}
strats = {'Never Query':never_query, 'Random Query':random_query, "Smart Query":smart_query}


def experiment(args):
    global time
    global rand_time
    results = {}
    results['graph 1'] = {}
    def rand_path_perm(stn_pos, worker_pos):
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

        random.shuffle(worker_path)
        return worker_path
    def dist(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    for strat in strats:
        results['graph 1'][strat] = []
    results['graph 1']['baseline'] = []
    for t in range(args.num_exp):
        print(f"Exp: {t}")
        if args.cluster_stations:
            num_clusters = args.num_stations//4
            cluster_pos = [(i,j) for i in range(args.grid_size//2) for j in range(args.grid_size//2)]
            cluster_pos = random.sample(cluster_pos, num_clusters)
            stations_pos = np.array([k for i,j in cluster_pos for k in [(2*i,2*j), (2*i+1, 2*j), (2*i, 2*j+1), (2*i+1, 2*j+1)]])
        else:
            stations_pos = [(i,j) for i in range(args.grid_size) for j in range(args.grid_size)]
            stations_pos = np.array(random.sample(stations_pos, args.num_stations))
        pos = [(i,j) for i in range(args.grid_size) for j in range(args.grid_size)]
        tool_box_pos = random.sample(pos, args.num_tool_locations)
        tools_pos = np.array([random.choice(tool_box_pos) for _ in range(args.num_stations)])
        fetcher_pos = np.array(random.choice(pos))
        worker_pos = np.array(random.choice(pos))
        goal = int(random.random()*len(stations_pos))
        env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stations_pos, tools_pos, goal, width=args.grid_size, height=args.grid_size)
        path = rand_path_perm(stations_pos[goal], worker_pos)
        results['graph 1']['baseline'].append(-int(max(dist(worker_pos, stations_pos[goal]), dist(fetcher_pos, tools_pos[goal])+dist(tools_pos[goal], stations_pos[goal]))))
        for strat in strats:
            print(f"Strat: {strat}")
            obs = env.reset()
            done = [False, False]
            fetcher = FetcherAltPolicy(query_policy=strats[strat])
            worker = PlanPolicy(path + [ToolFetchingEnvironment.WORKER_ACTIONS.WORK])
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
    parser.add_argument('--num_tool_locations', type=int, default=5, help='Number of toolboxes')

    args = parser.parse_args()

    results = experiment(args)
    with open(args.output, 'w') as f:
        json.dump(results, f)


