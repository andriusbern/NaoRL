
from nao_rl.learning import PPO
import nao_rl
import nao_rl.settings as s
import itertools, copy, time, datetime, pickle, json
import matplotlib.pyplot as plt
import numpy as np


def grid_search():
    """
    Script for trying all combinations of parameters specified 
    in the 'parameters dictionary'
    """

    # PARAMETERS

    number_of_repeats = 2 # Number of iterations per each combination of params

    parameters = {'env_name'       : ['nao-bipedal2'],
                  'n_workers'      : [4],
                  'max_episodes'   : [50],
                  'episode_length' : [100],
                  'batch_size'     : [128],
                  'epochs'         : [10],
                  'epsilon'        : [.15],
                  'gamma'          : [.99],
                  'actor_layers'   : [[200, 200], [300,300]],
                  'critic_layers'  : [[100, 100]],
                  'actor_lr'       : [.00001],
                  'critic_lr'      : [.00001]}
    

    values = tuple(parameters.values())
    param_iterator = list(itertools.product(*values))
    data = []
    counter = 0
    for params in param_iterator:
        counter += 1
        episode = []
        args = dict(zip(parameters.keys(), params))
        for i in range(number_of_repeats):
            print "\nIteration {} of parameter set {}/{}\nParameters:".format(i+1, counter, len(param_iterator))
            print args
            nao_rl.destroy_instances()
            time.sleep(.5)
            model = PPO(**args)
            model.train()

            episode.append(np.array(model.running_reward))
            
            
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
    with open(filename, 'ab') as f:
        f.writelines('\n')
        for arg in args:
            f.writelines('\n' + arg)


if __name__ == "__main__":
    data = grid_search()
    d = [[data[j][i] for j in range(len(data))] for i in range(len(data[0]))] 
    plt.plot(d)
    plt.show()