#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 02:32:32 2024

@author: batin13
"""



import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple
from torch.distributions import Normal
from torch import tensor,normal,randn_like,sqrt,zeros_like



gpu = 0

env = gym.make('Pendulum-v1')


if gpu:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
def tt(ndarray): 
    return Variable(torch.from_numpy(ndarray).float().to(device), requires_grad=False)


def clip_grad(model, clip_value):

    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = torch.clamp(param.grad.data, -clip_value, clip_value)


# class ActorNoise:
#     def __init__(self, mu, sigma, theta=0.15, dt=1e-2):
#         self.mu = mu
#         self.sigma = sigma
#         self.theta = theta
#         self.dt = dt
#         self.x_prev = np.zeros_like(self.mu)  # Ensuring x_prev is always initialized

#     def normal_noise(self, action):
#         noise = np.random.normal(self.mu, self.sigma, np.shape(action))
#         return action + noise
    
#     def ou_noise(self, action):
#         x_next = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
#                  self.sigma * np.sqrt(self.dt) * np.random.normal(size=np.shape(action))
#         self.x_prev = x_next  
#         return action + x_next
    
#     def ou_reset(self):
#         self.x_prev = np.zeros_like(self.mu)

            
    
        


class ActorNoise:
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_prev = zeros_like(tensor(self.mu))  # Ensuring x_prev is always initialized

    def normal_noise(self, action):
        noise = normal(self.mu, self.sigma, size=action.shape)
        return action + noise
    
    def ou_noise(self, action):
        x_next = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                 self.sigma * sqrt(tensor(self.dt)) * randn_like(action)
        self.x_prev = x_next  
        return action + x_next
    
    def ou_reset(self):
        self.x_prev = zeros_like(tt(self.mu))




class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim,hidden_dim, min_log_std,max_log_std,action_scale,mu_noise=0,sigma_noise=0.1,theta_noise=0.15,dt_noise=1e-2):
        super(Actor,self).__init__()
        
        self.min_std = min_log_std
        self.max_std = max_log_std
        
        self.action_scale = action_scale
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.non_linearity = F.relu6
        
        self.mu_net     = nn.Linear(hidden_dim, action_dim)
        self.log_std_net = nn.Linear(hidden_dim,action_dim)
        
        
        self.mu_noise = mu_noise
        self.sigma_noise = sigma_noise    
        self.theta_noise = theta_noise
        self.dt_noise = dt_noise
        
        self.noise = ActorNoise(self.mu_noise, self.sigma_noise,self.theta_noise, self.dt_noise)
        
    
    def forward(self,s):
        
        s = self.non_linearity(self.fc1(s))
        s = self.non_linearity(self.fc2(s))
        
        mu = self.mu_net(s)
        log_std = self.log_std_net(s)
        
        log_std = torch.clamp(log_std, self.min_std, self.max_std)
        
        return mu,log_std
        
    def sample(self,s):
        
        mean,log_std = self.forward(s)
        std = log_std.exp() 
        
        normal_dist = Normal(mean,std)
        x_t = normal_dist.rsample() 
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale 
        log_prob = normal_dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=1, keepdim=True)

        
        return action, log_prob
    
    def noisy_sample(self,s):

        
        action,log_prob = self.sample(s)
        
        noisy_action = self.noise.normal_noise(action)
        
        return noisy_action,log_prob
    
    def ou_noisy_sample(self,s):
        
        action , log_prob = self.sample(s)
        noisy_action = self.noise.ou_noise(action)
        
        return noisy_action, log_prob
    
    def reset_noise(self):
        self.noise.ou_reset()
        
        
    
    
    
class Critic(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim):
        super(Critic,self).__init__()
        
        input_dim = state_dim + action_dim
        
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)
        self.nl1 = F.relu6
        
        
    def forward(self,s,a):

        x = torch.cat([s,a],1)
        x1 = self.nl1(self.fc1(x))
        x1 = self.nl1(self.fc2(x1))
        x1 = self.fc3(x1)
        return x1
    
    
    
    
        
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        

        
        
        
        