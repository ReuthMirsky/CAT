import json
import pandas as pd
import argparse
from scipy import stats

def ttests(args):
    with open(args.file, 'r') as f:
        data = json.load(f)
    d = {}
    for graph in data:
        for k in data[graph].keys():
            if k == 'baseline':
                continue
            d[k] = {}
            for k2 in data[graph].keys():
                if k2 == 'baseline':
                    continue
                d[k][k2] = stats.ttest_rel(data[graph][k], data[graph][k2])
        df = pd.DataFrame(d)
        if args.output:
            df.to_csv(args.output)
        else:
            print(df)
                    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to run ttests on')
    parser.add_argument('-o', '--output', help='output file')
    args = parser.parse_args()
    ttests(args)
