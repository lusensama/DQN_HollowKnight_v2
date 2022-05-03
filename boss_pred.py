# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:28:51 2022

@author: yaoxinyi
"""

import cluster

pred = []

def start_predict(sc):
    skill_pred = cluster.get_boss_pred(sc)
    pred.append(skill_pred.cpu().data)
    
    