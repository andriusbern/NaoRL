from nao_rl.learning import models
import nao_rl.settings as s

def experiment(env, 
               n_steps,
               n_experiments,
               actor_layers,
               critic_layers,
               l_rate,
               gamma,
               random_process,
               logfile ):

    pass

    model = models.build_ddpg_model(env, actor_layers, critic_layers, gamma, l_rate)
    