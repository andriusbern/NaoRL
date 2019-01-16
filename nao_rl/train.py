"""
Author: Andrius Bernatavicius, 2018

Script for training RL Algorithms on the custom nao_rl or OpenAI gym environments

Arguments:
    Environments:
        1. [--NaoTracking]
        2. [--NaoBalancing] 
        3. [--NaoWalking] 
        4. Any OpenAI gym environment (e.g. 'BipedalWalker-v2')
    Algorithms:
        1. [--ppo] Proximal Policy Optimization (PPO) 
        2. [--a3c] Asynchronous Advantage Actor Critic (A3C)

Optional Arguments:
    1. Number of parallel workers (should not exceed the number of available cores):
        [--n_workers N] ---- e.g. [1:N] 

    2. Max number of episodes (depends on the difficulty of the environment):
        [--episodes N]  ---- e.g. [1e5:1e7]

    3. Max number of steps in each episode:
        [--steps N] 

    4. Number of layers and nodes in the actor network
        [--n_layers_actor N,N,N...] 

    5. Number of layers and nodes in the critic network
        [--n_layers_critic N,N,N...]

    6. Gamma (future reward discount factor)
        [--gamma N] [0:1]

    7. 
    

    Example to train with optimal parameters:
        'python train.py --NaoTracking --ppo'
    Custom parameters:
        'python train.py --NaoTracking --ppo --n_workers 4 --episodes 10000 --steps 500 --n_layers_actor 250 250 --n_layers_critic 250 250'
      *OR*
        edit the settings.py file that contains the default parameters
"""

import datetime, copy, time
import tensorflow as tf
import nao_rl
import gym
from argparse import ArgumentParser
from nao_rl.learning import PPO, A3C
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    start = time.time()
    parser = ArgumentParser() # Parse command line arguments
    
    
    # Default environment parameters

    algorithm = 'a3c'
    env_name = 'NaoTracking'
    render = True
    parameters = nao_rl.settings.default_parameters['{}_{}'.format(algorithm, env_name)]

    if algorithm == 'ppo':
        model = PPO(env_name, render, **parameters) #, **parameters
    if algorithm == 'a3c':
        model = A3C(env_name, render, **parameters)

    model.train()
    
      # Plot
    plt.plot(model.running_reward)
    plt.show()

    model.save()
    
    # model_path = '{}/{}_{}_{}.cpkt'.format(nao_rl.settings.TRAINED_MODELS, env_name, algorithm, date)
    # saver = tf.train.Saver()  # For saving models
    # saver.save(model.sess, model_path)
    # print 'Trained model saved at {}'.format(model_path)

    # Create a training log
    log = parameters.copy()
    log['env'] = env_name
    log['normalized_reward'] = model.running_reward
    log['episode_reward'] = model.episode_reward
    log['date'] = date 
    
    log_path = '{}/{}_{}_{}.log'.format(nao_rl.settings.DATA, env_name, algorithm, date)
    with open(log_path, 'w') as logfile:
        logfile.write(json.dumps(log))
    print 'Log file saved at {}'.format(log_path)


  