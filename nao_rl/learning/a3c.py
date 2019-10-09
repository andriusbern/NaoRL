"""
Threaded implementation of Asynchronous Advantage Actor Critic (A3C).
"""

import datetime
import threading, os
import tensorflow as tf
import numpy as np
import nao_rl
import gym
import time
import matplotlib.pyplot as plt


class A3C(object):
    def __init__(self, env_name, render, plot, n_workers=1, max_episodes=10000, episode_length=500,
                 update_every=10, entropy_beta=.005, gamma=.99,
                 actor_layers=[500,300], critic_layers=[500, 300],
                 actor_lr=.00005, critic_lr=.0001):

        # Training parameters
        self.gamma          = gamma
        self.beta           = entropy_beta
        self.max_episodes   = max_episodes
        self.episode_length = episode_length
        self.update_every   = update_every
        self.n_workers      = n_workers
        self.actor_layers   = actor_layers
        self.critic_layers  = critic_layers
        self.actor_lr       = actor_lr
        self.critic_lr      = critic_lr
        
        # Synchronization
        self.algorithm       = 'a3c'
        self.env_name        = env_name
        self.stop            = False
        self.total_steps     = 0
        self.update_counter  = 0
        self.current_episode = 0
        self.running_reward  = []
        self.episode_reward  = []
        self.time            = None
        self.verbose         = True
        self.date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        # Rendering
        if render == 0:
            self.render = [True for _ in range(self.n_workers)]
        if render == 1:
            self.render = [True for _ in range(self.n_workers)]
            self.render[0] = False
        if render == 2:
            self.render = [False for _ in range(self.n_workers)]

        # Plotting
        self.plot = plot
        if self.plot:
            plt.ion() 
            plt.figure(1)
            plt.plot()
            plt.xlabel('Episode')
            plt.ylabel('Running reward')
            plt.title('{} episode reward'.format(self.env_name))

         # Session and coordinator
        self.sess             = tf.Session()
        self.tf_coordinator   = tf.train.Coordinator()
        self.optimizer_actor  = tf.train.RMSPropOptimizer(self.actor_lr, name='RMSPropA')
        self.optimizer_critic = tf.train.RMSPropOptimizer(self.critic_lr, name='RMSPropC')
        self.workers = []

        # Environment parameters
        print "Creating dummy environment to obtain the parameters..."
        try:
            env = nao_rl.make(env_name, headless=True)
        except:
            env = gym.make(env_name)
        self.n_states  = env.observation_space.shape[0]
        self.n_actions  = env.action_space.shape[0]
        self.action_bounds = [env.action_space.low, env.action_space.high]
        nao_rl.destroy_instances()
        del env

        self.initialize()


    def initialize(self):
        """
        Create global network, workers and their networks, optimizers
        """
        with tf.device("/cpu:0"):
            self.global_net = ActorCriticNet('Global_net', self)
            self.create_workers()
            self.optimizer_actor  = tf.train.RMSPropOptimizer(self.actor_lr, name='RMSPropA')
            self.optimizer_critic = tf.train.RMSPropOptimizer(self.critic_lr, name='RMSPropC')


    def create_workers(self):
        """
        Initialize environments
        """
        for i in range(self.n_workers):
            print "\nCreating worker #{}...".format(i+1)
            try:
                env = nao_rl.make(self.env_name, headless=self.render[i])
            except:
                env = gym.make(self.env_name)
            worker = Worker(env, 'Worker_{}'.format(i+1), self)
            self.workers.append(worker)


    def train(self):

        self.sess.run(tf.global_variables_initializer())
        time.sleep(2)
        try:
            worker_threads = []
            for worker in self.workers:
                thread = threading.Thread(target=worker.work, args=())
                thread.start()
                worker_threads.append(thread)
            if self.plot:
                self.live_plot()
            self.tf_coordinator.join(worker_threads, ignore_live_threads=True, stop_grace_period_secs=5)
        except KeyboardInterrupt:
            print '\nStopped'
            self.stop = True
            self.tf_coordinator.should_stop()

            self.close_session()

            
    def save(self):
        model_path = '{}/{}_{}_{}.cpkt'.format(nao_rl.settings.TRAINED_MODELS, self.env_name, self.algorithm, self.date)
        saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=1)
        saver.save(self.sess, model_path)
        print 'Trained model saved at {}'.format(model_path)


    def close_session(self):
        for worker in self.workers:
            worker.env.disconnect()

    def summary(self):
        print 'Model summary:\n', \
              '--------------------\n',\
              'Environment: {}\n'.format(self.env_name),\
              'Parameters: {}\n'.format(locals())

    def live_plot(self):
        """
        Continuously plot the current running reward
        Must be run on the main thread
        """
        plot_count = 0
        while self.current_episode < self.max_episodes - 1 and not self.stop:
            if self.current_episode > 1 and self.current_episode > plot_count:
                plt.figure(1)
                plt.plot([plot_count, plot_count+1], self.running_reward[-2:], 'r')
                plt.plot([plot_count, plot_count+1], self.episode_reward[-2:], 'k', alpha=.2)
                plt.show()
                plt.pause(.0001)
                plot_count += 1

        plt.close()

    def plot_rewards(self):
        """
        Plot the whole reward history
        """
        plt.figure(2)
        plt.plot()
        plt.xlabel('Episode')
        plt.ylabel('Running reward')
        plt.title('{} episode reward'.format(self.env_name))
        plt.plot(self.running_reward)
        plt.plot(self.episode_reward)
        plt.show()

class ActorCriticNet(object):
    def __init__(self, scope, model, globalAC=None):
        
        self.model = model

        # Global parameters
        if scope == 'Global_net':  
            with tf.variable_scope(scope):
                self.state_input = tf.placeholder(tf.float32, [None, model.n_states], 'state_input')
                self._build_net()
                self.actor_parameters  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.critic_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        
        # Local parameters of workers
        else:   
            with tf.variable_scope(scope):
                self.state_input  = tf.placeholder(tf.float32, [None, model.n_states], 'state_input')
                self.action_input = tf.placeholder(tf.float32, [None, model.n_actions], 'action_input')
                self.v_target     = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v = self._build_net()

                # Temporal Difference
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('critic_loss'):
                    self.critic_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    self.test = sigma[0]
                    mu, sigma = mu * model.action_bounds[1], sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('actor_loss'):
                    log_prob        = normal_dist.log_prob(self.action_input)
                    expected_value  = log_prob * td
                    entropy         = normal_dist.entropy() # Add entropy
                    self.exp_v      = self.model.beta * entropy + expected_value
                    self.actor_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_action'):  # use local params to choose action
                    self.choose_action = tf.clip_by_value(tf.squeeze(normal_dist.sample(1)), *model.action_bounds, name='choose_action')
                with tf.name_scope('local_gradients'):
                    self.actor_parameters  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.critic_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.actor_gradients   = tf.gradients(self.actor_loss, self.actor_parameters)
                    self.critic_gradients  = tf.gradients(self.critic_loss, self.critic_parameters)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_actor_parameters  = [l_p.assign(g_p) for l_p, g_p in zip(self.actor_parameters, self.model.global_net.actor_parameters)]
                    self.pull_critic_parameters = [l_p.assign(g_p) for l_p, g_p in zip(self.critic_parameters, self.model.global_net.critic_parameters)]
                with tf.name_scope('push'):
                    self.update_actor  = self.model.optimizer_actor.apply_gradients(zip(self.actor_gradients, self.model.global_net.actor_parameters))
                    self.update_critic = self.model.optimizer_critic.apply_gradients(zip(self.critic_gradients, self.model.global_net.critic_parameters))

    def _build_net(self):
        """
        Build actor and critic networks
        """

        # Actor
        weight_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('actor'):
            hidden_layer = tf.layers.dense(self.state_input, self.model.actor_layers[0], tf.nn.relu6, kernel_initializer=weight_init, name='hl_actor0')
            for i, layer_size in enumerate(self.model.actor_layers[1::]):
                hidden_layer = tf.layers.dense(hidden_layer, layer_size, tf.nn.relu6, kernel_initializer=weight_init, name='hl_actor'+str(i+1))
            
            mu = tf.layers.dense(hidden_layer, self.model.n_actions, tf.nn.tanh, kernel_initializer=weight_init, name='mu')
            sigma = tf.layers.dense(hidden_layer, self.model.n_actions, tf.nn.softplus, kernel_initializer=weight_init, name='sigma')

        # Critic
        with tf.variable_scope('critic'):
            hidden_layer = tf.layers.dense(self.state_input, self.model.critic_layers[0], tf.nn.relu6, kernel_initializer=weight_init, name='hl_critic0')
            for i, layer_size in enumerate(self.model.critic_layers[1::]):
                hidden_layer = tf.layers.dense(hidden_layer, layer_size, tf.nn.relu6, kernel_initializer=weight_init, name='hl_critic'+str(i+1))
            
            value = tf.layers.dense(hidden_layer, 1, kernel_initializer=weight_init, name='v')  # state value

        return mu, sigma, value

    def update_global(self, feed_dict): 
        """
        Update the parameters of global actor-critic Net with the gradients collected from the workers
        """ 
        _, _, t = self.model.sess.run([self.update_actor, self.update_critic, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):
        """
        Get the trainable parameters of the global Actor-critic net
        """
        self.model.sess.run([self.pull_actor_parameters, self.pull_critic_parameters])

    def action(self, state):
        """
        Given a state, produce an action vector based on current policy
        """
        state = state[np.newaxis, :]
        return self.model.sess.run(self.choose_action, {self.state_input: state})


class Worker(object):
    def __init__(self, env, name, globalAC):

        self.env = env
        self.name = name
        self.model = globalAC
        self.local_net = ActorCriticNet(str(self.name), globalAC)

    def work(self):

        total_step = 1
        buffer_states, buffer_actions, buffer_rewards = [], [], []
        while not self.model.tf_coordinator.should_stop() and self.model.current_episode < self.model.max_episodes:
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            while True:
                action = self.local_net.action(state)
                state_, reward, done, info = self.env.step(action)
                if reward <= -100: reward=-2
                episode_reward += reward
                buffer_states.append(state)
                buffer_actions.append(action)
                buffer_rewards.append(reward)

                # Max episode length
                if episode_steps > self.model.episode_length:
                    done = True

                if total_step % self.model.update_every == 0 or done:   # update global and assign to local net
                    if done:
                        value_state_ = 0
                    else:
                        # Estimate value
                        value_state_ = self.model.sess.run(self.local_net.v, {self.local_net.state_input: state_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_rewards[::-1]:    # reverse buffer r
                        value_state_ = reward + self.model.gamma * value_state_
                        buffer_v_target.append(value_state_)
                    buffer_v_target.reverse()
                    
                    # Feed the buffer contents to the pipeline
                    buffer_states, buffer_actions, buffer_v_target = np.vstack(buffer_states), np.vstack(buffer_actions), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.local_net.state_input: buffer_states,
                        self.local_net.action_input: buffer_actions,
                        self.local_net.v_target: buffer_v_target,
                    }

                    test = self.local_net.update_global(feed_dict)
                    buffer_states, buffer_actions, buffer_rewards = [], [], [] # Clear buffer
                    self.local_net.pull_global()

                state = state_
                total_step += 1
                episode_steps += 1
                if done:
                    average_reward = episode_reward / float(episode_steps)
                    self.model.episode_reward.append(episode_reward)
                    if len(self.model.running_reward) == 0:  # record running episode reward
                        self.model.running_reward.append(episode_reward)
                    else:
                        self.model.running_reward.append(0.95 * self.model.running_reward[-1] + 0.05 * episode_reward)
                    print(
                        self.name,
                        "Ep:", self.model.current_episode,
                        "| Average reward: %.1f" % self.model.running_reward[-1],
                        '| Episode Reward: %.1f' % episode_reward,
                        '| Steps: %i' % episode_steps,
                        '| Average Reward: %.3f' % average_reward,
                    )
                    
                    self.model.current_episode += 1
                    
                    break

if __name__ == "__main__":
    model = A3C('NaoTracking', True, n_workers=1, max_episodes=5000, episode_length=500,
                 update_every=8, entropy_beta=.02, gamma=.99,
                 actor_layers=[50,50], critic_layers=[50],
                 actor_lr=.0001, critic_lr=.0002)

    # model.initialize()
    model.train()
