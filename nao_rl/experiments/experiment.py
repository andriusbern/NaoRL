
from nao_rl.learning import PPO
import nao_rl
import nao_rl.settings as s
import itertools, copy, time, datetime, pickle, json
import matplotlib.pyplot as plt
import numpy as np

"""
Environment ideas:
    Reward functions:

    1. With absolute position as only reward
    2. With feet position as only reward
    3. Both combined
"""

def grid_search():
    """
    Script for trying all combinations of parameters specified
    """
    # PARAMETERS
    number_of_repeats = 1 # Number of iterations per each combination of params

    parameters = {'env_name'       : ['NaoTracking'],
                  'n_workers'      : [4],
                  'max_episodes'   : [1000],
                  'episode_length' : [2000],
                  'batch_size'     : [1000, 500, 2000],
                  'epochs'         : [8],
                  'epsilon'        : [.2],
                  'gamma'          : [.99],
                  'actor_layers'   : [[50, 50], [100, 100]],
                  'critic_layers'  : [[50, 50]],
                  'actor_lr'       : [.00001],
                  'critic_lr'      : [.00002]}
    

    values = tuple(parameters.values())
    param_iterator = list(itertools.product(*values))
    data = []
    counter = 0
    for params in param_iterator:
        counter += 1
        args = dict(zip(parameters.keys(), params))
        for i in range(number_of_repeats):
            print "\nIteration {} of parameter set {}/{}\nParameters:".format(i+1, counter, len(param_iterator))
            print args
            nao_rl.destroy_instances()
            time.sleep(.5)
            model = PPO(**args)
            model.train()

            date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            filename = s.MAIN_DIR + '/data/' + parameters['env_name'][0] + '_' + date + '.log'
            log = args.copy()
            log['iteration'] = i
            log['exp_number'] = '{}/{}'.format(counter, len(param_iterator))
            log['global_reward'] = model.running_reward
            log['episode_reward'] = model.episode_reward
            log['date'] = date
            log['model_path'] = ''
            data.append(model.running_reward)

            model.close_session()
            del model

            with open(filename, 'w') as logfile:
                logfile.write(json.dumps(log))

    return data


def write(filename, *args):
    with open(filename, 'ab') as file:
        file.writelines('\n')
        for arg in args:
            file.writelines('\n' + arg)


if __name__ == "__main__":
    data = grid_search()
    d = [[data[j][i] for j in range(len(data))] for i in range(len(data[0]))] 
    plt.plot(d)
    plt.show()