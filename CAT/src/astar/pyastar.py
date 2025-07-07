import ctypes
import numpy as np
from profilehooks import profile
import inspect
from os.path import abspath, dirname, join

fname = abspath(inspect.getfile(inspect.currentframe()))
lib = ctypes.cdll.LoadLibrary(join(dirname(fname), 'astar.so'))

astar_c = lib.astar
ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
ndmat_i_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
astar_c.restype = ctypes.c_bool
astar_c.argtypes = [ndmat_i_type, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_int, ctypes.c_bool,
                  ndmat_i_type]

class astar():

    def __init__(self,start_pos,goal_pos,obstacles_pos,grid_size,interactive=False,**kwargs):
        start = start_pos
        goal=goal_pos
        dim=(grid_size,grid_size)
        obstacle_loc = obstacles_pos
        # Ensure goal is within bounds.
        if (goal[0] < 0 or goal[0] >= dim[0] or goal[1] < 0 or goal[1] >= dim[1]):
            raise ValueError('Goal of (%d, %d) lies outside grid.' % (goal))

        height, width = dim[0],dim[1]
        try:
            start_idx = np.ravel_multi_index(start, (height, width))
            goal_idx = np.ravel_multi_index(goal, (height, width))
        except:
            print(start,goal,height,width)
            raise Exception()

        if len(obstacle_loc) is 0:
            obstacle_locs_idx = np.array([],dtype=np.int32)
            n_locs = 0
        else:
            obstacle_locs_idx = np.ravel_multi_index(np.array(obstacle_loc),(height,width))
            obstacle_locs_idx = obstacle_locs_idx.astype('int32')
#             try:
#                 obstacle_locs_idx = np.ravel_multi_index(np.array(obstacle_loc).T,(height,width))
#                 obstacle_locs_idx = obstacle_locs_idx.astype('int32')
#             except ValueError:
#                 print("DEBUG FAIL")
#                 raise ValueError
            n_locs = len(obstacle_loc)

        #FOR now, we are assuming only 4 actions.
        allow_diagonal = False
        # The C++ code writes the solution to the paths array
        paths = np.full(height * width, -1, dtype=np.int32)
        success = astar_c(
            obstacle_locs_idx.flatten(),n_locs, height, width, start_idx, goal_idx, allow_diagonal,
            paths  # output parameter
        )

        self.found = success

        if not success:
            self.res = np.array([])
            return

        coordinates = []
        path_idx = goal_idx
        while path_idx != start_idx:
            pi, pj = np.unravel_index(path_idx, (height, width))
            coordinates.append((pi, pj))
            path_idx = paths[path_idx]

        if coordinates:
            coordinates.append(np.unravel_index(start_idx, (height, width)))
            self.res = np.vstack(coordinates[::-1])
            return
        else:
            self.res = np.array([])
            return

    def find_minimumpath(self):
        #return(is_found,retraced_path)
        #raise NotImplementedError
        return (self.found,self.res)
