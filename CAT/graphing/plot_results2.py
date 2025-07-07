"""
Plots output of experimental results file
"""
import context
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def graph(args):
    means = {}
    std = {}
    labels = []
    for fname in args.data:
        with open(fname, 'r') as f:
            results = json.load(f)
        for i,graph in enumerate(results):
            if graph != 'graph 1': continue
            if graph=='time':continue
            data = results[graph]
        if args.use_baseline:
            for k in data:
                if k == 'baseline': continue
                for i in range(len(data[k])):
                    data[k][i] -= data['baseline'][i]
        for k in data:
            if k == 'baseline': continue
            if k in args.skip: continue
            if k not in means:
                means[k] = []
                std[k] = []
                labels.append(k)
            #means[k].append(-1*np.mean(data[k]))
            means[k].append(np.mean(data[k]))
            std[k].append(np.std(data[k]))
    if args.labels:
        labels = args.labels
    if args.x_vals:
        x_vals = args.x_vals
    else:
        k = list(means.keys())[0]
        x_vals = list(range(len(means[k])))
    for k in means:
        print(k)
        print(len(x_vals))
        print(len(means[k]))
        assert len(x_vals) == len(means[k])
        assert len(x_vals) == len(std[k])

    plt.rcParams.update({'font.size':32})
    plt.rcParams.update({'pdf.fonttype':42})
    plt.rcParams.update({'ps.fonttype':42})
    if args.y_lims:
        plt.ylim(args.y_lims[0], args.y_lims[1])
    for k in means:
        plt.errorbar(x_vals, means[k],linewidth=5)
    plt.legend(labels)
    if args.x_axis:
        plt.xlabel(args.x_axis)
    if args.y_axis:
        plt.ylabel(args.y_axis)
    if args.title:
        plt.title(args.title)
    plt.show()

        

    #plt.rcParams.update({'font.size':8})
    #fig, ax = plt.subplots(len(list(results.keys())), 1, figsize=(15,10))

    #for i,graph in enumerate(results):
    #    data = results[graph]
    #    if args.use_baseline:
    #        for k in data:
    #            if k == 'baseline': continue
    #            for i in range(len(data[k])):
    #                data[k][i] -= data['baseline'][i]
    #    for k in data:
    #        if k == 'baseline': continue
    #        if k in args.skip: continue
    #        for j in range(len(data[k])):
    #            data[k][j] *= -1
    #    if len(results) > 1:
    #        axs = ax[i]
    #    else:
    #        axs = ax
    #    avg = {k:sum(data[k])/len(data[k]) for k in data}
    #    err_pos = {}
    #    err_neg = {}
    #    for k in data:
    #        if k == 'baseline':
    #            continue
    #        if k in args.skip:
    #            continue
    #        err_p = []
    #        err_n = []
    #        mu = avg[k]
    #        for p in data[k]:
    #            if p > mu:
    #                err_p.append(p - mu)
    #            else:
    #                err_n.append(mu - p)
    #        if len(err_p): 
    #            err_pos[k] = sum(err_p) / len(err_p)
    #        else:
    #            err_pos[k] = 0

    #        if len(err_n):
    #            err_neg[k] = sum(err_n) / len(err_n)
    #        else:
    #            err_neg[k] = 0

    #    worst_case = {k:max(data[k]) for k in data}
    #    
    #    labels = list(k for k in data.keys() if k != 'baseline' and k not in args.skip)

    #    positions = list(range(len(labels)))

    #    axs.errorbar(positions, [worst_case[labels[i]] for i in positions], fmt='ro')
    #    axs.errorbar(positions, [avg[labels[i]] for i in positions],
    #            [[err_neg[labels[i]] for i in positions], [err_pos[labels[i]] for i in positions]], fmt='o')
    #    axs.set_xticks(positions)
    #    axs.set_xticklabels(labels)
    #if args.output:
    #    plt.savefig(args.output)
    #else:
    #    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+', default=['results.json'], help='datafile to graph')
    parser.add_argument('--use_baseline', action='store_true', help='Flag should be present if baseline is subtracted from results')
    parser.add_argument('--skip', default = [], nargs='+', help='labels to skip graphing')
    parser.add_argument('--x_vals',  nargs='+', type=float, help='x-axis values per results file')
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('--x_axis', help="x-axis label")
    parser.add_argument('--y_axis', help="y-axis label")
    parser.add_argument('--labels', nargs='+', help='legend labels')
    parser.add_argument('--title', help='Figure Title')
    parser.add_argument("--y_lims", nargs=2, type=float, help="Y-axis limits")
    args = parser.parse_args()
    graph(args)




