# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:31:49 2019

@author: joousope
"""

import pygame
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
# %%
WIDTH=360
HEIGTH=360
FRAME=30

# %% colors

WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)
BLUE=(0,0,255)
GREEN=(0,255,0)
# DQL AGENT

class DQLAgent:
    
    def __init__(self):
        # hyper parameter
        self.state_size=4 #(distances player-m1 , player m2 = ((x1,x2),(y1,y2)),((x1,x3),(y1,y3)))
        
        self.action_size=3 # right left default(stay-none)
        
        self.gamma=0.95
        
        self.learning_rate=0.001
        
        self.epsilon=1 # explore 
        
        self.min_epsilon=0.1 
        
        self.epsilon_decay=0.995
        
        self.memory=deque(maxlen=1000)
        
        self.model=self.buildModel()
        #define parameter
    
    def buildModel(self):
        # neural network for deep neural network
        model=Sequential()
        model.add(Dense(48, input_dim=self.state_size,activation="relu"))
        model.add(Dense(self.action_size,activation="linear"))
        model.compile(loss="mse",optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self,state,action,reward,next_state,done):
        #storage
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self,state):
        state=np.array(state)
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        act_values=self.model.predict(state)
        return np.argmax(act_values[0])
            
    
    def replay(self,batch_size):
        if len(self.memory)<batch_size:
            return
        mini_batch=random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in mini_batch:
            state=np.array(state)
            next_state=np.array(next_state)
            if done:
                target=reward
            else:
                target=reward+self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target=self.model.predict(state)
            train_target[0][action]=target
            self.model.fit(state,train_target,verbose=0)
            
            
    def adaptiveEpsilonGreedy(self):
        if self.epsilon>self.min_epsilon:
            self.epsilon=self.epsilon*self.epsilon_decay
#        while self.epsilon>self.min_epsilon:
            

# Enemy class

class Enemy(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.Surface((10,10))
        self.image.fill(RED)
        self.rect=self.image.get_rect()
        self.radius=4
        pygame.draw.circle(self.image,WHITE,self.rect.center,self.radius)
        self.rect.x=random.randrange(0,WIDTH-self.rect.width)
        self.rect.y=random.randrange(2,6)
        self.speed_x=0
        self.speed_y=3
    def update(self):
        self.rect.x+=self.speed_x
        self.rect.y+=self.speed_y
        if(self.rect.top>HEIGTH+10):
            self.rect.x=random.randrange(0,WIDTH-self.rect.width)
            self.rect.y=random.randrange(2,6)
            self.speed_y=3
    def get_coordinate(self):
        return (self.rect.x,self.rect.y)       
# %% Player Class

class Player(pygame.sprite.Sprite):
    # sprite for the player
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.Surface((20,20))
        self.image.fill(BLUE)
        self.rect=self.image.get_rect()
        self.radius=10
        pygame.draw.circle(self.image,RED,self.rect.center,self.radius)
        self.rect.centerx=WIDTH/2
        self.rect.bottom=HEIGTH-1
#        self.y_speed=5
        self.x_speed=0
        
    def update(self,action):
        keystate=pygame.key.get_pressed()
        if keystate[pygame.K_LEFT] or action==0:
            self.x_speed=-4
        elif keystate[pygame.K_RIGHT] or action==1:
            self.x_speed=4
        else:
            self.x_speed=0
        self.rect.x+=self.x_speed
        if self.rect.right>WIDTH:
            self.rect.right=WIDTH
        elif self.rect.left<0:
            self.rect.left=0
    def get_coordinate(self):
        return (self.rect.x,self.rect.y)

# %%       
class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite=pygame.sprite.Group()
        self.enemy=pygame.sprite.Group()
        self.player=Player()
        self.all_sprite.add(self.player)
        self.m1=Enemy()
        self.m2=Enemy()
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        
        self.reward=0
        self.done=False
        self.total_reward=0
        self.agent=DQLAgent()
    def findDistance(self,a,b):
        d=a-b
        return d
    def step(self,action):
        state_list=[]
        
        # update
        self.player.update(action)
        self.enemy.update()
        
        # get coordinate
        next_player_state=self.player.get_coordinate()
        next_m1_state=self.m1.get_coordinate()
        next_m2_state=self.m2.get_coordinate()
        
        # find distances
        state_list.append(self.findDistance(next_player_state[0],next_m1_state[0]))
        state_list.append(self.findDistance(next_player_state[1],next_m1_state[1]))
        state_list.append(self.findDistance(next_player_state[0],next_m2_state[0]))
        state_list.append(self.findDistance(next_player_state[1],next_m2_state[1]))
        
        return [state_list]                  
    # reset
    def initialStates(self):
        self.all_sprite=pygame.sprite.Group()
        self.enemy=pygame.sprite.Group()
        self.player=Player()
        self.all_sprite.add(self.player)
        self.m1=Enemy()
        self.m2=Enemy()
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        
        self.reward=0
        self.done=False
        self.total_reward=0
        
        state_list=[]
        #get coordinate
        player_state=self.player.get_coordinate()
        m1_state=self.m1.get_coordinate()
        m2_state=self.player.get_coordinate()
        
        # find distances
        state_list.append(self.findDistance(player_state[0],m1_state[0]))
        state_list.append(self.findDistance(player_state[1],m1_state[1]))
        state_list.append(self.findDistance(player_state[0],m2_state[0]))
        state_list.append(self.findDistance(player_state[1],m2_state[1]))
        
        return [state_list]
    def run(self):
        #Game LOOP
        state=self.initialStates()
        
        running=True
        batch_size=24
        while running:
            self.reward=2
            clock.tick(FRAME)
            # process input
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running=False
            # update
            action=self.agent.act(state)
            next_state=self.step(action)
            self.total_reward+=self.reward
            hits=pygame.sprite.spritecollide(self.player,self.enemy,False,pygame.sprite.collide_circle)
            if hits:
                self.reward=-150
                self.total_reward+=self.reward
                self.done=True
                running=False
                print("Total Reward :{} ,".format(self.total_reward))
            self.agent.remember(state,action,self.reward,next_state,self.done)
            
            # update state
            state=next_state
            
            # training
            self.agent.replay(batch_size)
            
            # epsilon greddy
            self.agent.adaptiveEpsilonGreedy()
            
            #draw/render
            screen.fill(GREEN)
            self.all_sprite.draw(screen)

            #after drawing flip display
            pygame.display.flip() 
        pygame.quit()

# %% game loop
if __name__=="__main__":
    env=Env()
    liste=[]
    t=0
    while True:
        t+=1
        print("Episode: ",t)
        liste.append(env.total_reward)
        pygame.init()
        screen=pygame.display.set_mode((WIDTH,HEIGTH))
        pygame.display.set_caption("GAME")
        clock=pygame.time.Clock()
        
        env.run()
        