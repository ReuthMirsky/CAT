import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import os


def graph(args):
    results = {}
    for n in os.listdir(args.data):
        with open(args.data+'/'+n, 'r') as f:
            results[n] = json.load(f)

    c = args.cost
    dist = args.distribution


    data = []
    data2 = []

    key = None
    num_bins = 0

    count = 0
    counts = []
    lengthes = []
    for v in results:
        if c not in v:
            continue
        if dist not in v:
            continue
        print(v)
        key = v
        labels = list(results[key]['fetcher_actions'].keys())
        #data2 += list(results[v]['time']['Smart Query 3 No Random'])
        data2 += list(results[v]['time']['Best Query'])
        for ind, actions in enumerate(results[v]['fetcher_actions']['Best Query']):
        #for ind, actions in enumerate(results[v]['fetcher_actions']['Smart Query 3 No Random']):
            a = np.array(actions)
            temp = np.array(results[v]['action_times']['Best Query'][ind])
            for i in range(1, len(temp)):
                temp[i] -= temp[i-1]
            data += list(temp[np.where(a == 5)[0]])
            data += list(temp[np.where(a == 4)[0]])
            #temp2 = np.array(results[v]['queries_asked']['Best Query'])
            #temp3 = temp2[np.where(a == 5)[0]]
            #lengthes += [len(t) for t in temp3]
            count += len(np.where(a == 5)[0])
            counts.append(len(np.where(a == 5)[0]))
    print(np.mean(data2))
    print(np.max(data2))
    print("action times", np.mean(data))
    print(np.std(data))
    print(np.mean(lengthes))
    print(np.std(lengthes))
    print(count)
    print(np.mean(counts))
    print(np.std(counts))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help = 'data directory')
    parser.add_argument('-c', '--cost', help='Cost to use')
    parser.add_argument('-d', '--distribution', help='Distribution to use')
    args = parser.parse_args()
    graph(args)

            
