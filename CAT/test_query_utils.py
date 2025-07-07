from src import query_utils as qu
import numpy as np

goals = [(10,10), (0,0)]
grid = np.zeros((20,20))
toolbox = (5,5)
worker = (15,5)
fetcher = (0,0)

print(qu.get_ZQ(goals, grid, toolbox, worker, fetcher))
