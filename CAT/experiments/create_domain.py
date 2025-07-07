"""
Runs experimental analaysis of query costs in complicated domains
"""
import context
import random
from src.environment import ToolFetchingEnvironment
import numpy as np
from src.agents.agent import RandomWorkerPolicy
from src.agents.agent import PlanPolicy
from src.agents.agent_adhoc_q import FetcherQueryPolicy
from src.agents.agent_adhoc_q import FetcherAltPolicy
from src.agents.agent_adhoc_q import never_query, random_query, max_action_query, min_action_query, median_action_query, smart_query, smart_query2, create_smart_query3, smart_query_noRandom, smart_query2_noRandom, create_smart_query3_noRandom, create_optimal_query
from itertools import permutations
import pandas as pd
import json
import argparse
from time import sleep
from src.acd_utils import ACD2, WCD
import pickle

#strats = {'Never Query':never_query, 'Random Query':random_query, "Max Action Query":max_action_query, "Min Action Query":min_action_query, "Median Action Query":median_action_query, "Smart Query":smart_query}





def create_domain(args):
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
    edp = ACD2(stations_pos, width=args.grid_size, height=args.grid_size)
    wcd_f = WCD(tools_pos, width=args.grid_size, height=args.grid_size)
    wcd_a = WCD(stations_pos, width=args.grid_size, height=args.grid_size)
    domain = {"width":args.grid_size, "height":args.grid_size, "station_pos":stations_pos, "tools_pos":tools_pos, "fetcher_pos":fetcher_pos, "worker_pos":worker_pos, "edp":edp, "wcd_f":wcd_f, "wcd_a":wcd_a}
    return domain


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='results.json', help='Output File')
    parser.add_argument('--num_exp', default=100, type=int, help='Number of experiments to run')
    parser.add_argument('--grid_size', default=50, type=int, help='Grid Size of environment')
    parser.add_argument('--num_stations', default=20, type=int, help='Number of stations in environment')
    parser.add_argument('--cluster_stations', action='store_true', help='Flag should be present if stations are clustered')
    parser.add_argument('--num_tool_locations', type=int, default=5, help='Number of toolboxes')
    parser.add_argument('--base_cost', default=0.5, type=float, help='Base cost of querying')

    args = parser.parse_args()

    domain = create_domain(args)
    with open(args.output, 'wb') as f:
        pickle.dump(domain, f)


