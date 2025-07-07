import context
from src.environment2 import BlockWorld4Teams
from src.agents.BW4T_agent import GreedyAgent, SmartAgent, create_optimal_query
import numpy as np
import argparse
import json
from copy import deepcopy
from src.acd_utils2 import ACD2, WCD



def exp(args):
    num_rooms = args.num_rooms
    num_blocks = args.num_blocks
    results = {}
    results['graph 1'] = {}
    results['graph 1']['default'] = []
    results['graph 1']['GR'] = []
    results['graph 1']['query'] = []
    results['graph 1']['baseline'] = []
    for exp_num in range(args.num_exp):
        print("Exp Num:",exp_num)
        drop_room = np.random.randint(num_rooms)
        locations = np.random.randint(num_rooms, size=num_blocks)
        for i, l in enumerate(locations):
            while locations[i] == drop_room:
                locations[i] = np.random.randint(num_rooms)
        order = np.arange(0, num_blocks)
        np.random.shuffle(order)
        num_agents = 3
        #agents = [GreedyAgent(), SmartAgent(np.ones(shape=(2, num_blocks))/num_blocks)]
        agents = [GreedyAgent(), GreedyAgent(), GreedyAgent()]
        env = BlockWorld4Teams(num_rooms, drop_room, num_blocks, deepcopy(locations), deepcopy(order), num_agents, agents)
        results['graph 1']['baseline'].append(2*np.sum(np.abs(drop_room-locations))/len(agents))
        state = env.reset()
        done = False
        time = 0
        print(drop_room)
        while not done:
            state, reward, done, _ = env.step()
            #print(state[0])
            #print(state[1])
            #print(state[2])
            #print(state[3])
            #input()
            time += reward
        results['graph 1']['default'].append(time)
        agents = [GreedyAgent(), GreedyAgent(), SmartAgent(np.ones(shape=(2, num_blocks))/num_blocks)]
        #agents = [GreedyAgent(), GreedyAgent()]
        env = BlockWorld4Teams(num_rooms, drop_room, num_blocks, deepcopy(locations), deepcopy(order), num_agents, agents)
        state = env.reset()
        done = False
        time = 0
        print(drop_room)
        while not done:
            state, reward, done, _ = env.step()
            #print(state[0])
            #print(state[1])
            #print(state[2])
            #print(state[3])
            #input()
            time += reward
        results['graph 1']['GR'].append(time)
        edp = ACD2(deepcopy(locations), width=num_rooms)
        wcd = WCD(deepcopy(locations), width=num_rooms)
        agents = [GreedyAgent(), GreedyAgent(), SmartAgent(np.ones(shape=(2, num_blocks))/num_blocks, query_policy=create_optimal_query(args.cost, args.base_cost, edp, wcd))]
        env = BlockWorld4Teams(num_rooms, drop_room, num_blocks, deepcopy(locations), deepcopy(order), num_agents, agents, basecost=args.base_cost, cost=args.cost)
        state = env.reset()
        done = False
        time = 0
        print(drop_room)
        while not done:
            state, reward, done, _ = env.step()
            #print(state[0])
            #print(state[1])
            #print(state[2])
            #print(state[3])
            #input()
            time += reward
        results['graph 1']['query'].append(time)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='results.json', help='Output File')
    parser.add_argument('--num_exp', default=100, type=int, help='Number of experiments to run')
    parser.add_argument('--num_rooms', default=50, help='number of rooms in block world')
    parser.add_argument('--num_blocks', default=50, help='number of blocks in block world')
    parser.add_argument('-c', '--cost', default=0.1, type=float, help='Cost of including a station in a query')
    parser.add_argument('--base_cost', default=0.5, type=float, help='Base cost of querying')
    args = parser.parse_args()
    results = exp(args)
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f)
