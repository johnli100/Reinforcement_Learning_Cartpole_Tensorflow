# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:57:50 2019

@author: lizhi
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

env = gym.make('CartPole-v0')
numEpisodes = 10000
gamma = 0.99
h1= 16
h2 = 8
lr0=0.00001

def fc(state,h1_size,h2_size):
    fc1 = tf.layers.dense(state, h1_size)
    fc2 = tf.layers.dense(fc1,h2_size)
    value = tf.layers.dense(fc2,2)
    return value


tf.reset_default_graph() 
with tf.device('gpu:0'):
   state = tf.placeholder(tf.float64,[None,4])
   G = tf.placeholder(tf.float64,[None,2])
   learning_rate = tf.placeholder(tf.float64,shape=())
   Q_hat = fc(state,h1,h2)
   delta = G - Q_hat
   cost = tf.reduce_mean(tf.square(delta))
   #train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
   train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
   
start = time.time()      
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    obs = env.reset()
    episodes_t = np.zeros(numEpisodes)
        
    for i in range(numEpisodes):
        obs = env.reset()
        obs_init = obs
        done = False
        eps = 1/(i+1)**0.33
        lr = lr0/eps
        t=0
        gs=[]
        states=[]
        
        while not done:
            t += 1
            if np.random.random()<eps:
                action = env.action_space.sample()
            else:
                qs = sess.run(Q_hat,feed_dict={state:np.reshape(obs,[-1,4])})
                action = np.argmax(qs)
            
            prev_state = obs                        
            obs,reward,done,_ = env.step(action)
            reward = [0,reward] if action else [reward,0]
            #if done:
            #    reward = -100
            
            test=sess.run(Q_hat,feed_dict={state:np.reshape(obs_init,[-1,4])})
            
            states.append(prev_state)
            q = sess.run(Q_hat,feed_dict={state:np.reshape(obs,[-1,4])})
            gs.append(reward + gamma * q)
            #print(i,t,action,len(gs),len(states),lr)
        
        sess.run(train,feed_dict={state:np.reshape(states,[-1,4]),G:np.reshape(gs,[-1,2]),learning_rate:lr})         
        episodes_t[i] = t
#       print(t)

end=time.time()
print('Elapse time %.2f' %(end-start))
plt.plot(episodes_t)        

