# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 22:36:16 2022

@author: chenj
"""

import numpy as np
import os 
import matplotlib.pyplot as plt
    
self_hp_all = []
boss_hp_all = []
used_time_all = []
filenames = ['record.txt','record_final.txt']
for filename in filenames:
    fight_results = []
    boss_hp_lose = []
    self_hp_win = []
    used_time = []
    rewards = []
    
    with open(filename) as f:
        lines = f.readlines()
        # drop the first line
        for line in lines[1:]:
            tokens = line.split(',')
            fight_results.append(int(tokens[1]))
            # if win the fight, calculate the self hp and used time
            if int(tokens[1]) == 1:
                self_hp_win.append(int(tokens[2]))
                
                if len(tokens) == 5:
                    used_time.append(float(tokens[4]))
                if len(tokens) == 6:
                    used_time.append(float(tokens[5]))
            # if lose the fight, calculate the boss hp 
            else:
                boss_hp_lose.append(int(tokens[3]))
            if len(tokens) == 6:
                rewards.append(float(tokens[4]))
    
    
    print('win rate: ',sum(fight_results)/len(fight_results))
    print('average self hp: ', np.mean(self_hp_win))
    print('average boss hp: ',np.mean(boss_hp_lose))
    print('average time used %.3f s'%np.mean(used_time))    
    
    if rewards:
        # draw the evolution of rewards by running average
        ave_rewards = []
        wind = 10
        for i in range(len(rewards) - wind):
            ave_rewards.append(np.mean(rewards[i:i+wind]))
        plt.plot(ave_rewards)
        plt.show()
        
    self_hp_all.append(self_hp_win)
    boss_hp_all.append(boss_hp_lose)
    used_time_all.append(used_time)

def draw_cdf(data,label):
    # getting data of the histogram
    count, bins_count = np.histogram(data, bins=10)
  
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
  
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
      
    # plotting PDF and CDF
    plt.plot(bins_count[1:], cdf, label=label)
    # plt.legend()
    # plt.grid()
    # plt.show()
    
# draw_cdf(self_hp_win)

# draw_cdf(boss_hp_lose)

# draw_cdf(used_time)

labels = ['baseline','our approach']
for i in range(len(self_hp_all)):
    draw_cdf(self_hp_all[i], labels[i])
plt.legend(fontsize=20)
plt.grid()
plt.show()

for i in range(len(boss_hp_all)):
    draw_cdf(boss_hp_all[i], labels[i])
plt.legend(fontsize=20)
plt.grid()
plt.show()

for i in range(len(used_time_all)):
    draw_cdf(used_time_all[i], labels[i])
plt.legend(fontsize=20)
plt.grid()
plt.show()


