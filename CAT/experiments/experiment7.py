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
import copy

def query_beginning(obs, agent):
    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    if np.max(agent.probs) >= 1:
        return None
    goals = [g for g,p in enumerate(agent.probs) if p > 0]
    return np.random.choice(goals,size=len(goals)//2, replace=False)

rand_time = -1
time = 0
def query_random(obs, agent):
    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    if np.max(agent.probs) >= 1-0.001:
        return None
    if time >= rand_time:
        goals = [g for g,p in enumerate(agent.probs) if p > 0]
        return np.random.choice(goals,size=len(goals)//2, replace=False)
    return None

strats = {'Never Query':never_query, 'Query at beginning':query_beginning, "Query at ZQ":random_query, "Query at Random Time":query_random}


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
    trajs = []
    envs = []
    plans=  []
    for t in range(args.num_exp):
        #print(f"Exp: {t}")
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
        print(fetcher_pos, worker_pos)
        goal = int(random.random()*len(stations_pos))
        env = ToolFetchingEnvironment(fetcher_pos, worker_pos, stations_pos, tools_pos, goal, width=args.grid_size, height=args.grid_size)
        envs.append(env)
        path = rand_path_perm(stations_pos[goal], worker_pos)
        plans.append(path)
        results['graph 1']['baseline'].append(-int(max(dist(worker_pos, stations_pos[goal]), dist(fetcher_pos, tools_pos[goal])+dist(tools_pos[goal], stations_pos[goal]))))
        for strat in strats:
            #print(strat)
            obs = env.reset()
            done = [False, False]
            fetcher = FetcherAltPolicy(query_policy=strats[strat])
            worker = PlanPolicy(path + [ToolFetchingEnvironment.WORKER_ACTIONS.WORK])
            cost = 0
            rand_time = int(random.random()*args.grid_size)
            time = 0
            traj = [copy.deepcopy(obs)]
            while not done[0]:
                if args.render:
                    env.render()
                    sleep(0.05)
                obs, reward, done, _ = env.step([worker(obs[0]), fetcher(obs[1])])
                traj.append(copy.deepcopy(obs))
                w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, goal = obs[0]
                #print(f_action)
                #print(w_action)
                #input()
                cost += reward[1]
                time += 1
            results['graph 1'][strat].append(int(cost))
            if strat == 'Query at ZQ':
                trajs.append(traj)
            env.close()
    wBaseline = [d-b for d,b in zip(results['graph 1']['Query at ZQ'], results['graph 1']['baseline'])]
    env = envs[np.argmin(wBaseline)]
    path = plans[np.argmin(wBaseline)]
    #print(len(trajs))
    #print(len(trajs[0]))
    #print(len(trajs[0][0]))
    #print(len(trajs[0][0][0]))
    #print(trajs[np.argmin(wBaseline)][0][0])
    #print(trajs[np.argmin(wBaseline)][1][0])
    traj = trajs[np.argmin(wBaseline)]
    t_pos = traj[0][0][3]
    s_pos = traj[0][0][2]
    goal = traj[0][0][-1]
    print(t_pos[goal])
    print(s_pos[goal])
    f_pos = [traj[i][0][1] for i in range(len(traj))]
    print(f_pos)
    f_action = [traj[i][0][6] for i in range(len(traj))]
    print(f_action)
    for strat in strats:
            obs = env.reset()
            done = [False, False]
            fetcher = FetcherAltPolicy(query_policy=strats[strat])
            cost = 0
            worker = PlanPolicy(path + [ToolFetchingEnvironment.WORKER_ACTIONS.WORK])
            rand_time = int(random.random()*args.grid_size)
            time = 0
            while not done[0]:
                env.render()
                sleep(0.05)
                obs, reward, done, _ = env.step([worker(obs[0]), fetcher(obs[1])])
                w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, goal = obs[0]
                #input()
                cost += reward[1]
                time += 1
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

    args = parser.parse_args()


    results = experiment(args)
    with open(args.output, 'w') as f:
        json.dump(results, f)

