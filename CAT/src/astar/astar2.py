import numpy as np
import time
import matplotlib.pyplot as plt
import copy
from src.environment import Point2D
import heapq
import math

# RED = np.array([255,0,0])
# BLUE = np.array([0,0,255])
# GREEN = np.array([0,255,0])
# YELLOW = np.array([255,255,0])
# BLACK = np.array([0,0,0])
# CYAN = np.array((224,255,255))

RED = "#FF0000"
BLUE = "#1E90FF"
GREEN = "#7FFF00"
YELLOW = "#FFFF00"
BLACK = "#696969"
LINEN = "#FAF0E6"
BROWN = "#8B4513"

#todo:
# Add a update_function. We don't have to create new objects everytime we
# want to change end_goal or start_goal or some changes in the arena.
#NOT NEEDED. The creation of objects doesn't take much time.

class pos_node():
    def __init__(self,loc,parent_node,goal_loc,is_obstacle=False):
        self.x = loc[0]
        self.y = loc[1]
        self.loc = Point2D(loc[0],loc[1])

        if is_obstacle:
            self.is_obstacle = True
            self.parent = None
            self.goal_loc = None
            self.g = -99999
            self.h = -99999
            self.f = -99999
            return
        else:
            self.parent = parent_node
            self.h = self.euclidean_dist(goal_loc)
            if parent_node is not None:
                self.g = self.parent.g + 1 #distance from start
            else:
                self.g = 0

            self.update_f()
            self.is_obstacle = False
            self.move = None
            return

    def __add__(self,other):
        x = self.x+other.x
        y = self.y+other.y
        return Point2D(x,y)

    def __sub__(self, other):
        "self-other"
        x = self.x - other.x
        y = self.y - other.y
        return Point2D(x,y)


    def __hash__(self):
        return (self.x,self.y).__hash__()

    def __eq__(self, other):
        if isinstance(other, pos_node) or isinstance(other,Point2D):
            if ((self.x==other.x)and(self.y==other.y)):
                return True
            else:
                return False
        elif isinstance(other,tuple):
            if self.x==other[0] and self.y==other[1]:
                return True
        else:
            raise Exception("Unfair comparision")


    def __ne__(self, other):
        return not self.__eq__(other)


    def manhattan_dist(self,other):
        diff = self - other
        return abs(diff[0])+abs(diff[1])

    def euclidean_dist(self,other):
        diff = self -other
        return diff[0]*diff[0] + diff[1]*diff[1]


    def update_f(self):
        self.f = self.g + 100*self.h
        return

    def __str__(self):
        main_str = ' loc: '+str(self.loc)+' | h: '+str(self.h)+' | g: '+str(self.g)+' | f: '+str(self.f)+' | is_obstacle '+str(self.is_obstacle)+' '
        if self.parent:
            return main_str+' | parent_loc: '+str(self.parent.loc)
        else:
            return main_str



class astar():
    def __init__(self,start_pos,goal_pos,obstacles_pos,grid_size,interactive=False,**kwargs):
        self.grid_size = grid_size

        self.start_loc = Point2D(start_pos[0],start_pos[1])
        self.goal_loc = Point2D(goal_pos[0],goal_pos[1])
        self.obstacle_loc = [Point2D(pos[0],pos[1]) for pos in obstacles_pos]

        self.movement_directions = kwargs.get('movement_directions',4)
        self.distance_metric = kwargs.get('distance_metric', 'euclidean')
        self.interactive = interactive

        self._counter = 0
        # self.manual_obstacles = manual_obstacles

        self.start_node = pos_node(self.start_loc,None,self.goal_loc,False)
        self.goal_node = pos_node(self.goal_loc,None,self.goal_loc,True)


        self.open_heap = [] #Initializing as a list but needs to be used as a heap.

        heapq.heappush(self.open_heap,(self.start_node.f,self._counter,self.start_node))
        self._counter+=1
        self.open_pos = []

        self.closed_set = set()
        for obstacle in self.obstacle_loc:
            self.closed_set.add(pos_node(obstacle,None,None,True))


        '''if self.interactive:
            plt.ion()
            self.fig,self.ax = plt.subplots()
            grid = np.mgrid[0:self.grid_size,0:self.grid_size]
            x,y= grid[0],grid[1]

            color_mat = [[LINEN for i in range(self.grid_size)] for j in range(self.grid_size)]
            self.color_matrix = np.array(color_mat)
            self.color_matrix[self.goal_pos[0],self.goal_pos[1]]=BLACK
            self.color_matrix[self.start_pos[0],self.start_pos[1]]=BLACK
            for obs in self.obstacle_pos:
                self.color_matrix[obs[0]][obs[1]]=BROWN

            self.size_matrix = np.ones((self.grid_size,self.grid_size))*6000
            self.sc = plt.scatter(x,y,c=self.color_matrix.flatten(),s=self.size_matrix)
            plt.draw()
        '''


    def select_children(self,node):
        """
        Selects all non-closed, non-obstacle nodes according to
        movement_directions
        :return: list of copied (new-objects) children.
        """
        #optimize
        child_locs = []
        for move in MOVES[:-1]:
            new_pos = node.loc+move
            if (new_pos[0]>=0 and new_pos[0]<=self.grid_size) and (new_pos[1]>=0 and new_pos[1]<=self.grid_size) and (new_pos not in self.closed_set):
                child_locs.append((new_pos,move))

        return child_locs

    def solve(self):
        """
        Main algorithmic body.
        :return: final curr_pos
        """
        while(True):
            #loop init stuff
            try:
                #print(len(self.open_heap))
                curr_node = heapq.heappop(self.open_heap)[2]
            except Exception as e:
                print(e)
                raise Exception
            self.closed_set.add(curr_node)
            #print("Examining {}".format(curr_node.loc))

            #Select all non-closed, non-obstacle nodes as children
            children_locs = self.select_children(curr_node)
            #Update
            children_nodes = []
            for (child_loc,move) in children_locs:
                child_node = pos_node(child_loc,curr_node,self.goal_loc,False)
                child_node.move = move
                children_nodes.append(child_node)

            '''
            if self.interactive:
                #Set the color of all open positions to green
                for node in self.open_list:
                    self.color_matrix[node.loc[0],node.loc[1]]=GREEN
                #set the color of all closed positions to blue
                for node in self.close_list:
                    self.color_matrix[node.loc[0],node.loc[1]]=BLUE
                #set the color of curr_position to red
                self.color_matrix[self.curr_node.loc[0],self.curr_node.loc[1]] = RED
                #set the color of children to yellow
                for node in self.children_array:
                    self.color_matrix[node.loc[0],node.loc[1]]= YELLOW
                self.sc.set_color(self.color_matrix.flatten())
                self.fig.canvas.draw_idle()
                plt.pause(.551)
            '''

            #Decision
            for child_node in children_nodes:
                # print("***")
                # print(child)
                # print("***")
                if (child_node.loc == self.goal_loc):
                    #Goal has been reached, set the current node as the parent of the goal and terminate.
                    self.goal_node.parent = curr_node

                    #for returning purposes
                    return (1,self.goal_node)

                else:
                    #well, we have to take one step closer now that the goal hasn't been reached.
                    break_flag = 0
                    for i,(f,_,open_node)in enumerate(self.open_heap):
                        #Is the child node in open_heap already placed by previous explorations?
                        if open_node==child_node:
                            #Yes, someone has already put it here.
                            if child_node.g > open_node.g:
                                #Is the path to a child through the current node
                                #worse than through whatever was its parent?
                                pass
                            else:
                                #The path to this child is better through whatever we are doing now.
                                open_node.parent = curr_node
                                open_node.g = child_node.g
                                open_node.move = child_node.move
                                open_node.update_f()
                                self.open_heap[i] = (open_node.f,self._counter,open_node)
                                self._counter+=1
                                heapq.heapify(self.open_heap)
                            #Found the child in the open_heap, break.
                            break_flag = 1
                            break

                    if break_flag!=1:
                        #We didn't find this node in the open heap, so just add it.
                        heapq.heappush(self.open_heap,(child_node.f,self._counter,child_node))
                        self._counter+=1

            #Loop control
            if len(self.open_heap)==0:
                #print("Length is zero")
                return (0,curr_node)
        raise Exception("Error in program")

    def retrace_path(self,end_node):
        path = []
        curr_node = end_node
        while curr_node!=self.start_node:
            path.append(curr_node)
            curr_node = curr_node.parent
        path.append(curr_node)
        path.reverse()
        return path

    def find_minimumpath(self):
        found,end_node = self.solve()
        if not found:
            return (0,self.retrace_path(end_node))
        else:
            #print("Path to goal found")
            return (1,self.retrace_path(end_node))

    def print_path(self,res):
        found,path = res
        print(" \n Begin "+str(self.start_loc[0])+' '+str(self.start_loc[1]),end=' ')
        print("End "+str(self.goal_loc[0])+' '+str(self.goal_loc[1]))
        if found:
            print("Path is found")
        else:
            print("Path isn't found")
        for node in path:
            print('{}{}->'.format(node.x,node.y),end=' ')
        return
