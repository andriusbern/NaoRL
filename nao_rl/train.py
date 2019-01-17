"""
Author: Andrius Bernatavicius, 2018

Script for training RL Algorithms on the custom nao_rl or OpenAI gym environments

"""

import datetime, copy, time
import argparse
import json
import matplotlib.pyplot as plt

from nao_rl.learning import PPO, A3C
import tensorflow as tf
import nao_rl
import gym

def create_log(model, parameters):
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
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


if __name__ == "__main__":
    
    start = time.time()
    parser = argparse.ArgumentParser() # Parse command line arguments
    parser.add_argument('environment', type=str, help='RL environment to run.')
    parser.add_argument('algorithm', type=str, help='RL algorithm to use', choices=['a3c', 'ppo'])
    parser.add_argument('render', type=int, help='Rendering mode: [0] - no rendering, [1] - render first worker, [2] - render all workers', choices=[0,1,2])
    parser.add_argument('-p', '--plot', action='store_true', help='Live plotting of rewards during training')
    parser.add_argument('-n', '--n_workers', type=int, help='Number of parallel workers')
    parser.add_argument('-e', '--max_episodes', type=int, help='Max number of episodes to train')
    parser.add_argument('-s', '--episode_length', type=int, help='Max number of steps per episode')
    parser.add_argument('-g', '--gamma', type=float, help='Future reward discount factor')
    parser.add_argument('-a', '--actor_lr', type=float, help='Actor net learning rate')
    parser.add_argument('-c', '--critic_lr', type=float, help='Critic net learning rate')
    args = parser.parse_args()

    # Default environment parameters
    env_name = args.environment
    algorithm = args.algorithm
    render = args.render
    plot = args.plot

    # Get default parameters
    parameters = nao_rl.settings.default_parameters['{}_{}'.format(algorithm, env_name)]

    # Replace default parameters if any were provided from the command line
    for arg, value in vars(args).iteritems():
        if str(arg) in parameters and value is not None:
            parameters[str(arg)] = value
    
    print "Training..."
    print "----------------"
    print "Environment: {}".format(env_name)
    print "Algorithm:   {}".format(algorithm)
    print "Hyperparameters:"
    for args, values in parameters.items():
        print "  {} = {}".format(args, values)
    print "----------------"

    if algorithm == 'ppo':
        model = PPO(env_name, render, plot, **parameters) #, **parameters
    if algorithm == 'a3c':
        model = A3C(env_name, render, plot, **parameters)

    # Train and save the model  
    model.train()
    print 'Time elapsed: {} minutes.'.format((time.time() - start)/60)
    model.save()

    # Plots and logs
    create_log(model, parameters)
    model.plot_rewards()

    raw_input('Press enter to exit...')

    plt.close(2)
    nao_rl.destroy_instances()


  