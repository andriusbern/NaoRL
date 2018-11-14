from nao_rl.learning import models
import nao_rl.settings as s


def grid_search():

    import itertools

    # Static parameters
    env_name = 'nao-bipedal2'
    n_workers = 4
    max_episodes = 1000

    # Names of tunable parameters (corresponding to keyword arguments for the Experiment class)
    actor_layers = [100, 200]
    critic_layers = [100, 200]
    actor_lr = [.2, .3]
    critic_lr = [.2, .3]
    epsilon = [.2, .3]

    param_iterator = itertools.product(actor_layers,
                                       critic_layers,
                                       actor_lr,
                                       critic_lr,
                                       epsilon)

    
    for params in param_iterator:
        results =    Trainer(env_name      =env_name,
                             n_workers     =n_workers,
                             max_episodes  =max_episodes,
                             actor_layers  =params[0],
                             critic_layers =params[1],
                             actor_lr      =params[2],
                             critic_lr     =params[3],
                            )

