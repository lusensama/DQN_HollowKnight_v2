# Readme
> This repository is created in response to CV course project requirement. CSE 586/EE 554 Computer Vision II (Spring 2022) in Pennsylvania State University. 

## Environment

- windows 10 (We use win32 API to operate the little knight and get screenshots)
- python 3.8.8
- python liberary: find in `requirments.txt`
- Hollow Knight 1.4.3.2. (important!)
- HP Bar mod for Hollow Knight (In order to get the boss hp to calculate the reward, please find the mod in `./hollow_knight_Data/`, and then copy the mod file to the game folder)
- CUDA and cudnn for tensorflow and pytorch

## Usage

- Now I only write train.py but not test.py (the file is just test some base functions not for model), you can write it by yourself if you get a good model.
- The saving file for the game can be found in 'save' folder, if you never played this game, please move `/save_file/user3.dat` into save folder (usually `C:\user\_username_\AppData\LocalLow\Team Cherry\Hollow Knight`)
- Adjust the game resolution to 1920*1080 
- Run `train.py` for DQN (tensorflow), `train_ac.py` for A2C (pytorch). Those files can run separately.  
- Keep the game window at the forefront 
- Let the little knight stand in front of the statue of the boss in the godhome
- Press `F1` to start trainning. (Also you can use `F1` to stop trainning)
- We have uploaded the collected data in the folder `process_data`, just run `process.py` to get our evaluation results

## Code structure
- Most training configuration is in `train.py` and `train_ac.py`
- `Agent.py` and `Agent_ac.py` get output actions from two different models, respectively
- `DQN.py` `ActorCritic.py` are the learning algorithms
- `Model.py` defines the model wrapper for DQN (A2C will not use it)
- `ReplayMemory.py` defines the experience pool for learning

- Files in `./Tool` are for other functions we may use
- `Actions` defines actions for little knight and restart game script
- `GetHp` help us get our hp, boss hp, soul and location(it may have some bugs, you can fix it by yourself)
- `SendKey` is the API we use to send keyboard event to windows system.
- `UserInput` is an useless file, which I used it to train my model manually.
- `WindowsAPI` is used to get screenshot of the game, and `key_check()` is used to check which key is pressed.
- `Helper` defines [Reward Jugment] fucntion, and other functions we may use

## Changes

- Add `cluster.py` in Tool
- Add `boss_predictor.pth.tar` in model
- Modified `FrameBuffer.py` change get_frame() function, incorporate boss states clustering
- Modified `GetHP.py` change the get_hornet_pic() function
- Modified `ReplayMemory.py`: add a default argument flag when initializing the class to differentiate a2c and DQN, use flag to select different return values, do no need any change in original DQN `train.py`, add flag argument in `train_ac.py`

- Add `train_ac.py`
- Add `ActorCritic.py`
- Add `FrameBuffer_ac.py`
- Add `Agent_ac.py`
- Modified `Actions.py`: add function update_action()



