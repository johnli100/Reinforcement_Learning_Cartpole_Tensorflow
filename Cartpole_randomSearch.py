# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:42:55 2019

@author: lizhi
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

env= gym.make('CartPole-v0')
numEpisoids  = 5000
episoids_t = np.zeros(numEpisoids)
t_best = 0
w_best = np.random.random(4)*2 -1

def get_action(s,w):
    return 1 if s.dot(w) > 0 else 0

for i in range(numEpisoids):
    alpha = i/numEpisoids
    obs=env.reset()
    w = alpha * w_best + (1-alpha)*(np.random.random(4)*2 -1) 
    t = 0
    done = False
    
    while not done and t<5000:
        obs,reward,done, info = env.step(get_action(obs,w))
        t +=1
    
    episoids_t[i] = t
    if t>t_best:
        w_best = w 
        t_best = t
    
plt.plot(episoids_t)
plt.show()        

env.observation_space.sample()

1/2**0.5