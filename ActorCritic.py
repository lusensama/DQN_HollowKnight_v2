# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:52:27 2022

@author: jjc7191
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
import torch.multiprocessing as mp

import time
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENTROPY_BETA = 0.01
CLIP_GRAD = 0.1

class ac(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ac, self).__init__()

        self.input_shape = input_shape
        self.num_actions = n_actions
        
        self.split_critic_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.split_actor_conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        pre_out_size = self.get_pre_out(self.input_shape)
        
        self.policy = nn.Sequential(
            nn.Linear(pre_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
            
        self.value = nn.Sequential(
            nn.Linear(pre_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def get_pre_out(self,shape):
        o = self.split_critic_conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        actor_pre_result = self.split_actor_conv(x)
        critic_pre_result = self.split_critic_conv(x)
        actor_pre_result = actor_pre_result.view(actor_pre_result.size(0), -1)
        critic_pre_result = critic_pre_result.view(critic_pre_result.size(0), -1)
        return self.policy(actor_pre_result),self.value(critic_pre_result)

class Operator():
    def __init__(self,input_shape,act_dim,move_dim,lr_rate=0.0001):
        self.move_model = ac(input_shape,move_dim).to(device) 
        self.act_model = ac(input_shape,act_dim).to(device)
        self.move_optimizer = optim.Adam(params=self.move_model.parameters(), lr=lr_rate)
        self.act_optimizer = optim.Adam(params=self.act_model.parameters(), lr=lr_rate)
        #convert the network's output to a probability distribution of actions
        self.sm = nn.Softmax(dim=1).to(device)
        
    def load_model(self):
        RESULT_PATH = "./model/"
        if os.path.exists(RESULT_PATH+"move_ac.pt"):
            print("load move model")
            self.move_model.load_state_dict(torch.load(RESULT_PATH+"move_ac.pt"))
        if os.path.exists(RESULT_PATH+"act_ac.pt"):
            print("load action model")
            self.act_model.load_state_dict(torch.load(RESULT_PATH+"act_ac.pt"))
        
    def save_model(self):
        RESULT_PATH = "./model/"
        torch.save(self.move_model.state_dict(),RESULT_PATH+"move_ac.pt")
        torch.save(self.act_model.state_dict(),RESULT_PATH+"act_ac.pt")
        print("save model to the disk")

    def predict(self, state):
        with torch.no_grad():
            state = torch.unsqueeze(state, 0).to(device)
            # pred_move_probs_v = self.sm(self.move_model(state))
            # pred_move_probs = pred_move_probs_v.data.numpy()[0]
            # pred_move = np.random.choice(len(pred_move_probs), p=pred_move_probs)
            pred_move,value = self.move_model(state)
            pred_move = self.sm(pred_move)
            
            # pred_act_probs_v = self.sm(self.act_model(state))
            # pred_act_probs = pred_act_probs_v.data.numpy()[0]
            # pred_act = np.random.choice(len(pred_act_probs), p=pred_act_probs)
            pred_act,value = self.act_model(state)
            pred_act = self.sm(pred_act)
            return pred_move.cpu(), pred_act.cpu()
    
    def train(self,model,optim,batch_states,batch_actions,batch_qvals):
        # update the parameters of the model
        states_v = torch.stack(batch_states).to(device)
        batch_actions_t = torch.LongTensor(batch_actions).to(device)
        batch_qvals_v = torch.FloatTensor(batch_qvals).to(device)

        optim.zero_grad()
        logits_v, value_v = model(states_v)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_qvals_v)

        log_prob_v = F.log_softmax(logits_v, dim=1)
        adv_v = batch_qvals_v - value_v.detach()
        log_prob_actions_v = adv_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

        # calculate policy gradients only
        loss_policy_v.backward(retain_graph=True)

        # apply entropy and value gradients
        loss_v = entropy_loss_v + loss_value_v
        loss_v.backward()
        nn_utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optim.step()
        # # get full loss
        # loss_v += loss_policy_v
        return loss_policy_v.detach().cpu(), entropy_loss_v.detach().cpu(), loss_value_v.detach().cpu()
        
    def move_learn(self,obs,actions,reward,next_obs,terminal):
        return self.train(self.move_model, self.move_optimizer, obs, actions, reward)
        
    def act_learn(self,obs,actions,reward,next_obs,terminal):
        return self.train(self.act_model, self.act_optimizer, obs, actions, reward)
        
        