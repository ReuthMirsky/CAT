"""
Plots output of experimental results file
"""
import context
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
plt.rcParams.update({'font.size':32})
plt.rcParams.update({'pdf.fonttype':42})
plt.rcParams.update({'ps.fonttype':42})
print(list(plt.rcParams.keys()))


def graph(args):
    with open(args.data, 'r') as f:
        results = json.load(f)

    plt.rcParams.update({'font.size':8})
    fig, ax = plt.subplots(len(list(results.keys())), 1, figsize=(15,10))

    for i,graph in enumerate(results):
        if graph != 'graph 1': continue
        data = results[graph]
        if args.use_baseline:
            for k in data:
                if k == 'baseline': continue
                for i in range(len(data[k])):
                    data[k][i] -= data['baseline'][i]
                print(k, data[k])
                plt.hist(data[k], bins=100)
                plt.show()
                input()
        for k in data:
            if k == 'baseline': continue
            if k in args.skip: continue
            for j in range(len(data[k])):
                #data[k][j] *= -1
                data[k][j] *= 1
        if len(results) > 1:
            axs = ax[i]
        else:
            axs = ax
        avg = {k:sum(data[k])/len(data[k]) for k in data}
        err_pos = {}
        err_neg = {}
        for k in data:
            if k == 'baseline':
                continue
            if k in args.skip:
                continue
            err_p = []
            err_n = []
            mu = avg[k]
            for p in data[k]:
                if p > mu:
                    err_p.append(p - mu)
                else:
                    err_n.append(mu - p)
            if len(err_p): 
                err_pos[k] = sum(err_p) / len(err_p)
            else:
                err_pos[k] = 0

            if len(err_n):
                err_neg[k] = sum(err_n) / len(err_n)
            else:
                err_neg[k] = 0

        worst_case = {k:max(data[k]) for k in data}
        
        labels = list(k for k in data.keys() if k != 'baseline' and k not in args.skip)

        positions = list(range(len(labels)))

        plt.rcParams.update({'font.size':32})
        plt.rcParams.update({'axes.labelsize':32})
        for item in axs.get_xticklabels() + axs.get_yticklabels():
            item.set_fontsize(32)
        axs.errorbar(positions, [worst_case[labels[i]] for i in positions], fmt='ro')
        axs.errorbar(positions, [avg[labels[i]] for i in positions],
                [[err_neg[labels[i]] for i in positions], [err_pos[labels[i]] for i in positions]], fmt='o')
        axs.set_xticks(positions)
        axs.set_xticklabels(labels)
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', default='results.json', help='datafile to graph')
    parser.add_argument('--use_baseline', action='store_true', help='Flag should be present if baseline is subtracted from results')
    parser.add_argument('--skip', default = [], nargs='+', help='labels to skip graphing')
    parser.add_argument('-o', '--output', help='output file')
    args = parser.parse_args()
    graph(args)




