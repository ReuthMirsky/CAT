import context
from src.environment import ToolFetchingEnvironment
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

    key = None
    num_bins = 0

    for v in results:
        if dist not in v:
            print(v)
            continue
        if c not in v:
            continue
        key = v
        labels = list(r for r in results[key]['fetcher_actions'].keys() if r not in args.skip)
        for k in labels:
            l = []
            print(k, len(results[v]['fetcher_actions'][k]))
            for actions in results[v]['fetcher_actions'][k]:
                a = np.array(actions)
                num_bins = max(num_bins, len(a))
                print(a)
                l += list(np.where(a == ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY)[0])
            data.append(l)
    print(data)
    d = np.array(data[1])
    print(np.where(d == 0))
    print(int(ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY))

    if args.labels:
        labels = args.labels


    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    fig.tight_layout()
    plt.rcParams.update({'font.size':27})
    #plt.rc('axes', titlesize=100)
    #plt.rc('axes', labelsize=100)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    #plt.tick_params(axis='both', which='major', labelsize=22)
#    plt.ylim(0, 80)
    #plt.xlim(left=-0.5, right=20)
    for i in range(len(data)):
        ax[i//2,i%2].hist(data[i], bins=num_bins+1, linewidth=1, histtype='bar')
        ax[i//2,i%2].set_title(labels[i])
        ax[i//2, i%2].set_ylim(0, 80)
        ax[i//2, i%2].set_xlim(-0.5, 30)
        ax[i//2, i%2].tick_params(axis='both', which='major', labelsize=27)
        #ax[i//2,i%2].rcParams.update({'font.size':32})
        if args.x_axis:
            ax[i//2, i%2].set_xlabel(args.x_axis, fontsize=15)
        if args.y_axis:
            ax[i//2,i%2].set_ylabel(args.y_axis, fontsize=15)
    if args.x_axis:
        plt.xlabel(args.x_axis)
    if args.y_axis:
        plt.ylabel(args.y_axis)
    if args.title:
        plt.title(args.title)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help = 'data directory')
    parser.add_argument('-c', '--cost', help='Cost to use')
    parser.add_argument('-d', '--distribution', help='Distribution to use')
    parser.add_argument('--skip', nargs='+', default=[], help='Methods to skip in Graph')
    parser.add_argument('--labels', nargs='+', help='Method labels')
    parser.add_argument('--x_axis', help='x-axis label')
    parser.add_argument('--y_axis', help='y-axis-label')
    parser.add_argument('--title', help='Figure Title')
    args = parser.parse_args()
    graph(args)

            

