# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import time
import collections
import matplotlib.pyplot as plt

from ActorCritic import ac,Operator
from Agent_ac import Agent
from ReplayMemory import ReplayMemory


import Tool.Helper
import Tool.Actions
from Tool.Helper import mean, is_end
from Tool.Actions import take_action, update_action, restart,take_direction
from Tool.WindowsAPI import grab_screen
from Tool.GetHP import Hp_getter
from Tool.UserInput import User
from Tool.FrameBuffer_ac import FrameBuffer

window_size = (0,0,1920,1017)
station_size = (230, 230, 1670, 930)

HP_WIDTH = 768
HP_HEIGHT = 407
WIDTH = 400
HEIGHT = 200
FRAMEBUFFERSIZE = 4
INPUT_SHAPE = (FRAMEBUFFERSIZE, HEIGHT, WIDTH)

MEMORY_SIZE = 600  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 24  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 10  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.0001  # 学习率
GAMMA = 0

action_name = ["Attack", "Attack_Up",
           "Short_Jump", "Mid_Jump", "Rush"]
ACTION_DIM = len(action_name)
update_action()

move_name = ["Move_Left", "Move_Right", "Turn_Left", "Turn_Right"]
MOVE_DIM = len(move_name)

DELAY_REWARD = 1




def run_episode(hp, model,agent,act_rmp_correct, move_rmp_correct,PASS_COUNT,paused):
    restart()
    loss_policy_move, entropy_loss_move, loss_value_move = [],[],[]
    loss_policy_act, entropy_loss_act, loss_value_act = [],[],[]
    # learn while load game
    for i in range(8):
        if (len(move_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("move learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = move_rmp_correct.sample(BATCH_SIZE)
            loss_1, loss_2, loss_3 = model.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)   
            loss_policy_move.append(loss_1)
            entropy_loss_move.append(loss_2)
            loss_value_move.append(loss_3)

        if (len(act_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("action learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp_correct.sample(BATCH_SIZE)
            loss_1, loss_2, loss_3 = model.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)
            loss_policy_act.append(loss_1)
            entropy_loss_act.append(loss_2)
            loss_value_act.append(loss_3)
    
    step = 0
    done = 0
    total_reward = 0

    start_time = time.time()
    # Delay Reward
    DelayMoveReward = collections.deque(maxlen=DELAY_REWARD)
    DelayActReward = collections.deque(maxlen=DELAY_REWARD)
    DelayStation = collections.deque(maxlen=DELAY_REWARD + 1) # 1 more for next_station
    DelayActions = collections.deque(maxlen=DELAY_REWARD)
    DelayDirection = collections.deque(maxlen=DELAY_REWARD)
    
    while True:
        self_hp = hp.get_self_hp()
        boss_hp_value = hp.get_boss_hp()
        if boss_hp_value > 800 and  boss_hp_value <= 900 and self_hp >= 1 and self_hp <= 9:
            break
        

    thread1 = FrameBuffer(1, "FrameBuffer", WIDTH, HEIGHT, maxlen=FRAMEBUFFERSIZE)
    thread1.start()

    last_hornet_y = 0
    while True:
        step += 1
        # last_time = time.time()
        # no more than 10 mins
        # if time.time() - start_time > 600:
        #     break

        # in case of do not collect enough frames
        
        while(len(thread1.buffer) < FRAMEBUFFERSIZE):
            time.sleep(0.1)
        
        stations = thread1.get_buffer()
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        player_x, player_y = hp.get_play_location()
        hornet_x, hornet_y = hp.get_hornet_location()
        soul = hp.get_souls()

        hornet_skill1 = False
        if last_hornet_y > 32 and last_hornet_y < 32.5 and hornet_y > 32 and hornet_y < 32.5:
            hornet_skill1 = True
        last_hornet_y = hornet_y

        move, action = agent.sample(stations, soul, hornet_x, hornet_y, player_x, hornet_skill1)

        
        # action = 0
        take_direction(move)
        take_action(action)
        
        # print(time.time() - start_time, " action: ", action_name[action])
        # start_time = time.time()
        
        next_station = thread1.get_buffer()
        next_boss_hp_value = hp.get_boss_hp()
        next_self_hp = hp.get_self_hp()
        next_player_x, next_player_y = hp.get_play_location()
        next_hornet_x, next_hornet_y = hp.get_hornet_location()

        # get reward
        move_reward = Tool.Helper.move_judge(self_hp, next_self_hp, player_x, next_player_x, hornet_x, next_hornet_x, move, hornet_skill1)
        # print(move_reward)
        act_reward, done = Tool.Helper.action_judge(boss_hp_value, next_boss_hp_value,self_hp, next_self_hp, next_player_x, next_hornet_x,next_hornet_x, action, hornet_skill1)
            # print(reward)
        # print( action_name[action], ", ", move_name[d], ", ", reward)
        
        DelayMoveReward.append(move_reward)
        DelayActReward.append(act_reward)
        DelayStation.append(stations)
        DelayActions.append(action)
        DelayDirection.append(move)

        if len(DelayStation) >= DELAY_REWARD + 1:
            if DelayMoveReward[0] != 0:
                move_rmp_correct.append((DelayStation[0],DelayDirection[0],DelayMoveReward[0],DelayStation[1],done))
            # if DelayMoveReward[0] <= 0:
            #     move_rmp_wrong.append((DelayStation[0],DelayDirection[0],DelayMoveReward[0],DelayStation[1],done))

        if len(DelayStation) >= DELAY_REWARD + 1:
            if mean(DelayActReward) != 0:
                act_rmp_correct.append((DelayStation[0],DelayActions[0],mean(DelayActReward),DelayStation[1],done))
            # if mean(DelayActReward) <= 0:
            #     act_rmp_wrong.append((DelayStation[0],DelayActions[0],mean(DelayActReward),DelayStation[1],done))

        station = next_station
        self_hp = next_self_hp
        boss_hp_value = next_boss_hp_value
            
        end_time = time.time()
        
        # if (len(act_rmp) > MEMORY_WARMUP_SIZE and int(step/ACTION_SEQ) % LEARN_FREQ == 0):
        #     print("action learning")
        #     batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp.sample(BATCH_SIZE)
        #     model.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)

        total_reward += act_reward
        paused = Tool.Helper.pause_game(paused)

        if done == 1:
            Tool.Actions.Nothing()
            break
        elif done == 2:
            PASS_COUNT += 1
            Tool.Actions.Nothing()
            time.sleep(3)
            break
        

    thread1.stop()

    for i in range(80):
        if (len(move_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("move learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = move_rmp_correct.sample(BATCH_SIZE)
            loss_1, loss_2, loss_3 = model.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)   
            loss_policy_move.append(loss_1)
            entropy_loss_move.append(loss_2)
            loss_value_move.append(loss_3)
        if (len(act_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("action learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp_correct.sample(BATCH_SIZE)
            loss_1, loss_2, loss_3 = model.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)
            loss_policy_act.append(loss_1)
            entropy_loss_act.append(loss_2)
            loss_value_act.append(loss_3)
    # if (len(move_rmp_wrong) > MEMORY_WARMUP_SIZE):
    #     # print("move learning")
    #     batch_station,batch_actions,batch_reward,batch_next_station,batch_done = move_rmp_wrong.sample(1)
    #     model.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)   

    # if (len(act_rmp_wrong) > MEMORY_WARMUP_SIZE):
    #     # print("action learning")
    #     batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp_wrong.sample(1)
    #     model.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)
    total_loss = np.array([loss_policy_move, entropy_loss_move, loss_value_move, loss_policy_act, entropy_loss_act, loss_value_act])
    return total_reward, step, PASS_COUNT, self_hp, boss_hp_value, end_time - start_time, total_loss


if __name__ == '__main__':

    # In case of out of memory
    # config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True      #程序按需申请内存
    # sess = tf.compat.v1.Session(config = config)

    
    total_remind_hp = 0

    act_rmp_correct = ReplayMemory(MEMORY_SIZE, file_name='./act_memory',flag=1)         # experience pool
    move_rmp_correct = ReplayMemory(MEMORY_SIZE,file_name='./move_memory',flag=1)         # experience pool
    
    # create a file to record the fighting
    if not os.path.exists('record.txt'):
        with open('record.txt','w') as f:
            f.write('episode, result, self hp, boss hp, total reward, time\n')
    if not os.path.exists('loss.txt'):
        with open('loss.txt','w') as f:
            f.write('loss_policy_move, entropy_loss_move, loss_value_move, loss_policy_act, entropy_loss_act, loss_value_act\n')    
    # new model, if exit save file, load it
    ac_model = Operator(INPUT_SHAPE, ACTION_DIM, MOVE_DIM, LEARNING_RATE)  

    # Hp counter
    hp = Hp_getter()


    ac_model.load_model()
    
    agent = Agent(ACTION_DIM,ac_model,e_greed=0.12,e_greed_decrement=1e-6)
    
    # get user input, no need anymore
    # user = User()

    # paused at the begining
    paused = True
    paused = Tool.Helper.pause_game(paused)

    max_episode = 30000
    # 开始训练
    episode = 0
    PASS_COUNT = 0           
    prev_count = 0                            # pass count
    while episode < max_episode:    # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        episode += 1     

        total_reward, total_step, PASS_COUNT, remain_self_hp, remain_boss_hp, time_period, total_loss = run_episode(hp, ac_model,agent,act_rmp_correct, move_rmp_correct, PASS_COUNT, paused)
        if episode % 5 == 0:
            ac_model.save_model()
        # if episode % 5 == 0:
        #     move_rmp_correct.save(move_rmp_correct.file_name)
        # if episode % 5 == 0:
        #     act_rmp_correct.save(act_rmp_correct.file_name)
        total_remind_hp += remain_self_hp
        print("Episode: ", episode, ", pass_count: " , PASS_COUNT, ", hp:", total_remind_hp / episode)

        # write the fighting results in this episode
        with open('record.txt','a') as f:
            f.write("%d, %d, %d, %d, %.3f, %.3f\n"%(episode,PASS_COUNT-prev_count,remain_self_hp,remain_boss_hp,total_reward,time_period))
        with open('loss.txt','a') as f:
            f.write("Episode: %d\n"%episode)
            for i in range(total_loss.shape[1]):
                f.write(",".join([str(j) for j in total_loss[:,i]])+"\n")
        prev_count = PASS_COUNT
