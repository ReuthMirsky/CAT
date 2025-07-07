import numpy as np
import matplotlib.pyplot as plt
import time

import pygame
import pygame.camera
import time
import threading
from colour import Color

def convert_to_rgb(col):
    r,g,b = col
    return (int(r*255),int(g*255),int(b*255))

RED = Color("red")
BLUE = Color

mild_blue = (88, 28, 255) #581CFF
harder_blue = (188,28,255)
mild_white_blue = (195, 174, 255)

fbg_agent = [mild_white_blue,mild_blue]
fbg_agent_adhoc = [mild_white_blue,harder_blue]

flgreen = (248, 255, 0)
almost_black = (48,48,41)
fbg_object = [almost_black,flgreen]


#What this code should expect

"""
Initial parameters.
1) Dims of the arena
2) The agents and their properties to display
    - Only agent numbers, and types.

Time-varying parameters
1) Location and orientation of each agent.
2) Presence/Absence of food

"""



class Visualizer():
    def __init__(self,dims,object_pos,agent_pos,agent_orientations):
        """
        A helper class to visualize the AAMAS paper.
        :param dims: Dimension of the square grid.
        :param agent_pos: list of agent positions to start with.
        :param agent_orientations: Orientations of the agents.
        :param object_pos: List of object positions to start with.
        """
        pygame.init()
        pygame.mixer.quit()
        self.agent_pos = agent_pos
        self.agents_orientations = agent_orientations
        self.object_pos = object_pos
        self.gridsize = (dims,dims)

        #Setting up display related parameters.
        self.box_size = 100 #Each of the boxes in the domain is 60px wide
        self.screen = pygame.display.set_mode((self.gridsize[0]*self.box_size,self.gridsize[1]*self.box_size))
        self.clock = pygame.time.Clock()
        self.screen.fill((255,255,255))
        self.done=False
        self.screen_memory = self.screen

        #setting up text related parameters.
        pygame.font.init()
        self.font = pygame.font.SysFont("comicsansms", 45)
        # /self.wait_on_event()
        self.update_event_type = pygame.USEREVENT+1
        # self.wait_on_event()
        return


    def visualize(self,obj_pos,agents_positions,agents_orientations):
        """
        :param obj_pos: List of object positions given in _point2d format
        :param agents_positions: List of agent positions given in _point2d format
        :param agents_orientations: List of agent orientations given in one of the 4 directions.
        :return:
        """
        self.agents_positions = agents_positions
        self.agents_orientations = agents_orientations

        self.screen.fill((255,255,255))
        self.draw_arena()
        self.draw_objects(obj_pos)
        self.draw_agents(agents_positions,agents_orientations)
        pygame.transform.flip(self.screen,False,True)

        pygame.display.flip()

    def draw_arena(self):
        for i in range(self.gridsize[0]):
            for j in range(self.gridsize[1]):
                x = j*self.box_size
                y = i*self.box_size
                color = (0,128,255) #Blue borders.
                pygame.draw.rect(self.screen,color,pygame.Rect(x,y,self.box_size,self.box_size),2)

    def draw_objects(self,object_pos_list):
        for i,object_pos in enumerate(object_pos_list):
            object_pos_inPixels = (object_pos.x * self.box_size, (self.gridsize[1] - object_pos.y) * self.box_size)

            x, y = object_pos_inPixels
            rect_arc = pygame.draw.rect(self.screen, fbg_object[1], pygame.Rect(x, y, self.box_size, self.box_size))
            text = self.font.render(str('O{}').format(i+1), True, fbg_object[0])
            self.screen.blit(text, (x + self.box_size // 2 - text.get_height() // 2, y + self.box_size // 2 - text.get_width() // 2))

    def draw_agents(self,agent_pos_list,agent_orientation_list):
        """
        :param agent_pos_list:
        :param agent_orientation_list: Orientation of agent as one of Up,Down,Left,Right, in that order.
        :return:
        """
        for i,agent_pos in enumerate(agent_pos_list):
            agent_pos_inPixels = (agent_pos.x * self.box_size, (self.gridsize[1]-agent_pos.y)*self.box_size)
            x,y = agent_pos_inPixels
            rect_arc = pygame.draw.circle(self.screen,fbg_agent[1],(x+self.box_size//2,y+self.box_size//2),self.box_size//2)
            text = self.font.render(str('A{}').format(i+1),True,fbg_agent[0])
            self.screen.blit(text,(x+self.box_size//2 -text.get_height()//2,y+self.box_size//2 - text.get_width()//2))

            orientation = agent_orientation_list[i]
            dir_dot_size = 3
            delta = 3
            if orientation==0:
                #UP
                pygame.draw.circle(self.screen,fbg_agent[0],[x+self.box_size//2,y+delta],dir_dot_size)
            elif orientation==1:
                #DOWN
                pygame.draw.circle(self.screen,fbg_agent[0],[x+self.box_size//2,y+self.box_size-delta],dir_dot_size)
            elif orientation==2:
                #RIGHT
                pygame.draw.circle(self.screen,fbg_agent[0],[x-delta+self.box_size,y+self.box_size//2],dir_dot_size)
            elif orientation==3:
                #LEFT
                pygame.draw.circle(self.screen,fbg_agent[0],[x+delta,y+self.box_size//2],dir_dot_size)
            else:
                raise Exception("Invalid direction in visualizer.")

    def wait_on_event(self):
        while not self.done:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                self.done = True
                pygame.display.quit()
                pygame.quit()
                break
            elif event.type == self.update_event_type:
                self.visualize(event.obj_positions,event.agents_positions,event.agents_orientations)

    def snapshot(self,name):
        pygame.image.save(self.screen,name+str('.png'))
        return




if __name__ == "__main__":
    import random
    from src.global_defs import _point2d as point2d
    n_agents = 4
    agent_positions_x = random.sample(range(20),4)
    agent_positions = [point2d(ele,ele) for ele in agent_positions_x]

    agent_orientations = random.sample(range(4),4)


    object_positions = [point2d(ele,ele+2) for ele in agent_positions_x]

    vis = Visualizer(20,object_positions,agent_positions,agent_orientations)
    vis.visualize(object_positions,agent_positions,agent_orientations)
