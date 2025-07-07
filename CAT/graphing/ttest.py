from scipy.stats import  ttest_rel
import argparse
import json


def ttest(args):
    with open(args.data, 'r') as f:
        results = json.load(f)
    for i,graph in enumerate(results):
        if graph != 'graph 1': continue
        print(f"Graph {i}: {graph}")
        data = results[graph]
        for k1 in data:
            if k1 == 'baseline': continue
            if k1 in args.skip: continue
            for k2 in data:
                if k2 == k1 or k2 == 'baseline': continue
                if k2 in args.skip: continue
                print(f"{k1} {k2} {ttest_rel(data[k1], data[k2])}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', default='results.json', help='datafile to graph')
    parser.add_argument('--skip', default = [], nargs='+', help='labels to skip graphing')
    args = parser.parse_args()
    ttest(args)
