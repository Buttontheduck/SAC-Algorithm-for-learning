#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:46:18 2024

@author: batin13
"""

import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch import tensor
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Networks_Pendulum import Actor, Critic, clip_grad,ActorNoise
from datetime import datetime
from collections import namedtuple
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import socket
import warnings
import time

wb = False  # Do you want to log your data to weighs and biases? True, activate weigths and biases
print(f'\n  Logging to Weights and biases: {wb}  \n')

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])
env = gym.make('Pendulum-v1')

gpu = 0

if gpu:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def check_internet_connection():
    try:
        # Try to establish a connection to the main server of Google.
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        return False
 
connect = check_internet_connection()
print(f'\n  INTERNET CONNECTION : {connect}  \n')



def set_global_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def soft_update(target,source,tau):
    for target_param, param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(target_param.data*( 1 - tau) + param.data * tau)

def hard_update(target,source):
    soft_update(target,source,1)

def tt(ndarray): 
    return Variable(torch.from_numpy(ndarray).float().to(device), requires_grad=False)
      
class Buffer:
    
        def __init__(self,max_size):
            self.max_size = max_size
            self.buffer = []

        def add_transition(self,tup):                   
            #state action next_state rewad terminal_flag
            if not len(tup) == 5:
                raise ValueError("Tuple length must be 5")

            if len(self.buffer)<self.max_size:
                self.buffer.append(tup)
            else:
                self.buffer.pop(0)
                self.buffer.append(tup)
                warnings.warn("Max buffer size is exceed, Therefore first element of the buffer is popped")
                
                    
        def get_transition(self,batch_size):
            current_len = len(self.buffer)
            
            if current_len < batch_size:
                raise ValueError("Length of the buffer is smaller than the batch size")
            
            
            rand_choice = np.random.choice(len(self.buffer),batch_size)
            
            batch_state      = []
            batch_next_state = []
            batch_reward     = []
            batch_action     = []
            batch_flag       = []
             
            
            for ind in rand_choice:
                batch_state.append(self.buffer[ind][0])
                batch_action.append(self.buffer[ind][1])
                batch_next_state.append(self.buffer[ind][2])
                batch_reward.append(self.buffer[ind][3])
                batch_flag.append(self.buffer[ind][4])
                
                
            return tt(np.array(batch_state)),tt(np.array(batch_action)),tt(np.array(batch_next_state)),tt(np.array(batch_reward)),tt(np.array(batch_flag))
        

        
class SoftActorCritic(nn.Module):
    
    def __init__(self,state_dim,action_dim,hidden_dim,gamma,tau,max_std,min_std,actor_learning_rate,critic_learning_rate,
                 alpha_learning_rate,buffer_max_size,update_together,update_freq,mu,sigma):
        super(SoftActorCritic,self).__init__()
        
        

        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.max_std = max_std
        self.min_std = min_std
        self.tau = tau
        self.ac_lr = actor_learning_rate
        self.cr_lr = critic_learning_rate  
        self.update_freq = update_freq
        self.update_together = update_together
        self.action_scale = 2
        
        self.mu = mu
        self.sigma = sigma

        
        self.actor =  Actor(self.state_dim, self.action_dim, self.hidden_dim, self.min_std, self.max_std,self.action_scale,self.mu,self.sigma)
        
        self.crit1 = Critic(self.state_dim, self.action_dim, self.hidden_dim)
        self.crit2 = Critic(self.state_dim, self.action_dim, self.hidden_dim)
        
        self.crit1_target = Critic(self.state_dim, self.action_dim, self.hidden_dim)
        self.crit2_target = Critic(self.state_dim, self.action_dim, self.hidden_dim)  
        
        
        hard_update(self.crit1_target, self.crit1)
        hard_update(self.crit2_target, self.crit2)
        
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape[0]).to(device)).item() 
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.optim_alpha = optim.Adam([self.log_alpha], lr=alpha_learning_rate)


        self.actor.to(device)
        self.crit1.to(device)
        self.crit2.to(device)

        self.crit1_target.to(device)
        self.crit2_target.to(device) 

        self.optim_actor   = optim.Adam(self.actor.parameters(), lr = self.ac_lr)
        self.optim_crit1   = optim.Adam(self.crit1.parameters(), lr = self.cr_lr) 
        self.optim_crit2   = optim.Adam(self.crit2.parameters(), lr = self.cr_lr) 
        
        self.loss_actor = nn.MSELoss()
        self.loss_state = nn.MSELoss()

        self.buffer_max_size = buffer_max_size
        
        self.memory = Buffer(self.buffer_max_size)
        
        
    def buffer_filling(self,batch_size):
        
        s  = env.reset()
        
        for i in range(batch_size*10):
            print(f"\rFilling {i+1}", end='')
            s = tt(s)
            a,log_prob = self.actor.sample(s)
            a = a.detach().numpy()
            ns,r,flag,_ = env.step(a)
            
            tup = (s,a,ns,r,flag)
            
            self.memory.add_transition(tup)
            
            if flag:
                s = env.reset()
            else:
                s = ns
        print("\nBUFFER is filled ")
        time.sleep(1)        
    
    def train(self,episodes,batch_size,timesteps):
        
        stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes))    
        
        
        self.buffer_filling(batch_size)
        
        
        for e in range(episodes):
            
            state = env.reset()
            for t in range(timesteps): 
                
                action, log_probs = self.actor.sample(tt(state))
                action = action.detach().numpy()
                next_state,reward,terminal_flag,_ = env.step(action)
                
                tup = (state,action,next_state,reward,terminal_flag)
                self.memory.add_transition( tup)
                if terminal_flag:
                    state = env.reset()
                else:
                    state = next_state
                    
                
                stats.episode_rewards[e] += reward
                stats.episode_lengths[e] = t
                
                render = True
                #if render and (e)%33 ==0 or (e)%34 ==0 or (e)%35 ==0 :
                if render and (e)%100==0 :
                    env.render()
                    time.sleep(0.01)
                    
                
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags  =   self.memory.get_transition(batch_size)
                
                with torch.no_grad():
                    next_action, next_log_prob = self.actor.noisy_sample(batch_next_states)
                    q1_next_target = self.crit1_target(batch_next_states, next_action)
                    q2_next_target = self.crit2_target(batch_next_states, next_action)
                    q_min_next_state = torch.min(q1_next_target, q2_next_target)
                
    
                alpha =  self.log_alpha.exp()
                q_error =  q_min_next_state - alpha * next_log_prob
                next_q_value =  batch_rewards.unsqueeze(1) + self.gamma * (1-batch_terminal_flags.unsqueeze(1)) * q_error.squeeze(0)

                current_q1 = self.crit1(batch_states,batch_actions)
                current_q2 = self.crit2(batch_states,batch_actions)  


                loss_q1 = self.loss_state(current_q1,next_q_value.detach())
                loss_q2 = self.loss_state(current_q2,next_q_value.detach())
            
                if e % self.update_freq==0:
                    self.update_soft = True
                else:
                    self.update_soft = False
                    
                #self.critic_params_update(loss_q1, loss_q2, self.update_soft , self.update_together)
                    
                self.update_crit1(loss_q1, self.update_soft)
                self.update_crit2(loss_q2, self.update_soft)       
                
                
                act, log_probs = self.actor.noisy_sample(batch_states)
                
                q1 = self.crit1(batch_states,act)
                q2 = self.crit2(batch_states,act)
                q_min = torch.min(q1,q2)
                
                actor_loss =  ((alpha * log_probs) - q_min).mean()
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                self.actor_params_update(actor_loss, alpha_loss)



            if wb and connect:

                wandb.log({
                                "episode reward": int(stats.episode_rewards[e]),
                                "episode length": int(stats.episode_lengths[e]),
                                "alpha value": alpha,
                                "actor loss": actor_loss,
                                "critic loss 1": loss_q1,
                                "critic loss 2": loss_q2
                                })

            elif wb and not connect and e<10:
                print("There is no internet connection")

            if e > 1:

                print("\r{}/{}  &  reward =  {} ".format(e+1, episodes, stats.episode_rewards[e]), end="")
         
                    
        return stats


               
                


    def critic_params_update(self,q1_loss ,q2_loss, update_soft , update_together = False):
        
        if update_together:
            
            q_loss = q1_loss + q2_loss
            
            self.optim_crit1.zero_grad()
            self.optim_crit2.zero_grad()
            
            q_loss.backward() 
            
            self.optim_crit1.step()
            self.optim_crit2.step()
            
        else:  
            
            self.optim_crit1.zero_grad()
            q1_loss.backward() 
            self.optim_crit1.step()
            
            self.optim_crit2.zero_grad()
            q2_loss.backward()
            self.optim_crit2.step()
            
        if update_soft:
            soft_update(self.crit1_target, self.crit1, self.tau)
            soft_update(self.crit2_target, self.crit2, self.tau)
        
    def actor_params_update(self,actor_loss,alpha_loss):
        
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
        
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()       
        
    def new_transition(self):
        state = env.reset()
        action, log_probs = self.actor.sample(tt(state))
        action = action.detach().numpy()
        next_state,reward,terminal_flag,_ = env.step(action)
        
        tup = (state,action,next_state,reward,terminal_flag)
        self.memory.add_transition( tup)
        return reward
    
    def update_crit1(self,loss,update_soft):
                    
        self.optim_crit1.zero_grad()
        loss.backward() 
        self.optim_crit1.step()
        if update_soft:
            soft_update(self.crit1_target, self.crit1, self.tau)
    
    def update_crit2(self,loss,update_soft):
                    
        self.optim_crit2.zero_grad()
        loss.backward() 
        self.optim_crit2.step()
        if update_soft:
            soft_update(self.crit2_target, self.crit2, self.tau)
    
              
        
if __name__ == "__main__":
    
    
 
    seeding = 1
    current_time = datetime.now().strftime('%d.%m.%y.%H.%M.%S')
    print(f'\nRun started {current_time}\n')
    env = gym.make('Pendulum-v1')
    
    env.seed(seeding)
    set_global_seeds(seeding)
    
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    hidden_dim = 32
    gamma = 0.99
    episodes = 10000
    time_steps = 200
    actor_learning_rate  = 0.001
    critic_learning_rate = 0.001
    alpha_learning_rate  = 0.001
    max_buffer_size = int(1e6)
    update_delay = 2
    batch_size = 64
    tau = 0.001
    max_log_std = 2
    min_log_std = -20
    crit_loss_update_together = False
    noise_mu = 0
    noise_sigma = 0.2
 

    if wb and connect:
        wandb.init(

        project="SAC Pendulum ",
        name = "run 1.1",

        config={
            "hidden_dim ": hidden_dim,
            "seed": seeding,
            "gamma ": gamma,
            "episodes ": episodes,
            "tau ": tau,
            "Update Delay ": update_delay,
            "Buffer Size":max_buffer_size,
            "time_steps ": time_steps,
            "actor learning rate  ": actor_learning_rate,
            "critic learning rate  ": critic_learning_rate,
            "alpha learning rate  ": alpha_learning_rate,
            "alpha": 0,
            "Maximum Log STD ": max_log_std,
            "Minimum Log STD ": min_log_std,            
            "batch size ": batch_size,
            "Actor noise mean ": noise_mu,            
            "Actor noise sigma ": noise_sigma,
            "Critic loss calculation added together": crit_loss_update_together


                })
        
    sac = SoftActorCritic(state_dim,action_dim,hidden_dim,gamma,tau,max_log_std,min_log_std,
                              actor_learning_rate,critic_learning_rate,alpha_learning_rate,max_buffer_size,
                              crit_loss_update_together,update_delay,noise_mu,noise_sigma)
        
    stats = sac.train(episodes,batch_size,time_steps)
        
        
        

        
        
        
        

        
