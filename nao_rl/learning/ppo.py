"""
Author: Andrius Bernatavicius, 2018

Threaded implementation of PPO
"""

import tensorflow as tf
import numpy as np
import gym, threading, queue
import nao_rl, time


class PPO(object):
    def __init__(self, env_name, n_workers=4,  max_episodes=5000, episode_length=500,
                 batch_size=128, epochs=10, epsilon=.2, gamma=.99,
                 actor_layers=[500,500], critic_layers=[500],
                 actor_lr=.00001, critic_lr=.00002):

        self.vrep_port = 19998

        # Training parameters
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_workers = n_workers
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        # Synchronization
        self.env_name = env_name
        self.total_steps = 0
        self.update_counter = 0
        self.current_episode = 0
        self.running_reward = []
        self.episode_reward = []
        self.time = None
        self.verbose = True

        # Threading and events
        self.sess = tf.Session()
        self.update_event = None
        self.rollout = None
        self.tf_coordinator = None
        self.queue = None

        # Environment parameters
        print "Creating dummy environment to obtain the parameters..."
        # nao_rl.destroy_instances()
        env = nao_rl.make(self.env_name, self.vrep_port)
        self.vrep_port -= 1
        self.action_space  = env.action_space.shape[0]
        self.state_space   = env.observation_space.shape[0]
        self.action_bounds = [env.action_space.low[0], -env.action_space.low[0]]
        env.disconnect()
        time.sleep(.5)
        nao_rl.destroy_instances()
        del env

        ##############
        ### Network ##
        ##############

        # Input placeholders
        self.state_input       = tf.placeholder(tf.float32, [None, self.state_space], 'state_input')
        self.action_input      = tf.placeholder(tf.float32, [None, self.action_space],'action_input')
        self.advantage_input   = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.discounted_reward = tf.placeholder(tf.float32, [None, 1], 'discounted_reward')

        ########
        # Critic
        hidden_layer = tf.layers.dense(self.state_input, critic_layers[0], tf.nn.relu) 
        for layer_size in critic_layers[1::]:
            hidden_layer = tf.layers.dense(hidden_layer, layer_size, tf.nn.relu)
        self.critic_output = tf.layers.dense(hidden_layer, 1)
        
        self.advantage = self.discounted_reward - self.critic_output
        self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
        self.critic_optimizer = tf.train.AdamOptimizer(critic_lr).minimize(self.critic_loss)

        #######
        # Actor
        policy, pi_params = self.build_actor('policy', True, actor_layers)
        old_policy, oldpi_params = self.build_actor('old_policy', False, actor_layers)
        self.choose_action = tf.squeeze(policy.sample(1), axis=0, name='choose_action')  
        self.update_policy = [old.assign(p) for p, old in zip(pi_params, oldpi_params)]
        ratio = policy.prob(self.action_input) / (old_policy.prob(self.action_input) + 1e-5)
        surrogate_loss = ratio * self.advantage_input

        # Clipped objective
        self.actor_loss = -tf.reduce_mean(tf.minimum(surrogate_loss,
                                                     tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * self.advantage_input))
        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss)
        self.sess.run(tf.global_variables_initializer())


    def build_actor(self, name, trainable, layers):
        """
        Build actor network
        """
        with tf.variable_scope(name):
            # Hidden layers
            hidden_layer = tf.layers.dense(self.state_input, layers[0], tf.nn.relu, trainable=trainable)
            for layer_size in layers[1::]:
                hidden_layer = tf.layers.dense(hidden_layer, layer_size, tf.nn.relu, trainable=trainable)
            # Output layer
            mu = 2 * tf.layers.dense(hidden_layer, self.action_space, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(hidden_layer, self.action_space, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma, name='policy')

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params


    def update(self):
        """
        Update the global network
        """
        while not self.tf_coordinator.should_stop():
            if self.current_episode < self.max_episodes:
                self.update_event.wait()
                # Copy policy
                self.sess.run(self.update_policy)
                data = [self.queue.get() for _ in range(self.queue.qsize())]   
                data = np.vstack(data)

                states  = data[:, :self.state_space]
                actions = data[:, self.state_space: self.state_space + self.action_space]
                rewards = data[:, -1:]
                advantage = self.sess.run(self.advantage, {self.state_input: states, self.discounted_reward: rewards})

                # Update actor and critic networks
                for _ in range(self.epochs):
                    self.sess.run(self.actor_optimizer,  {self.state_input: states, self.action_input: actions, self.advantage_input: advantage}) 
                    self.sess.run(self.critic_optimizer, {self.state_input: states, self.discounted_reward: rewards})

                self.update_event.clear()       
                self.update_counter = 0         
                self.rollout.set()        


    def action(self, state):
        """
        Pick an action based on state
        """
        state = state[np.newaxis, :]
        action = self.sess.run(self.choose_action, {self.state_input: state})[0]
        return np.clip(action, self.action_bounds[0], self.action_bounds[1])


    def get_critic_output(self, state):
        """
        Compute the value estimate
        """
        if state.ndim < 2: 
            state = state[np.newaxis, :]
        return self.sess.run(self.critic_output, {self.state_input: state})[0, 0]


    def create_workers(self):
        """
        Initialize environments
        """
        self.workers = []
        for i in range(self.n_workers):
            env = nao_rl.make(self.env_name, self.vrep_port, headless=True)
            self.vrep_port -= 1
            worker = Worker(env, self, i)
            self.workers.append(worker)

    def train(self):
        self.update_event, self.rollout = threading.Event(), threading.Event()
        self.tf_coordinator = tf.train.Coordinator()
        self.queue = queue.Queue()     
        self.rollout.set()     
        self.update_event.clear()          
        try:
            self.create_workers()
            threads = []
            for worker in self.workers:         
                t = threading.Thread(target=worker.work, args=())
                t.start()                 
                threads.append(t)
            threads.append(threading.Thread(target=self.update,))
            threads[-1].start()
            self.time = time.time()
            self.tf_coordinator.join(threads, ignore_live_threads=True, stop_grace_period_secs=5)
            #self.tf_coordinator.wait_for_stop()
        except KeyboardInterrupt:
            print "Training stopped..."
            # self.tf_coordinator.request_stop()
            # #self.tf_coordinator.wait_for_stop()
            # self.close_session()

        
    def close_session(self):
        for worker in self.workers:
            worker.env.disconnect()
        self.sess.close()
        tf.reset_default_graph()


class Worker(object):
    def __init__(self, env, global_ppo, worker_name):
        self.worker_name = env.port
        self.env = env
        self.trainer = global_ppo

    def work(self):
        while not self.trainer.tf_coordinator.should_stop():
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            state_buffer, action_buffer, reward_buffer = [], [], []
            for t in range(self.trainer.episode_length):
                if not self.trainer.rollout.is_set() and not self.trainer.tf_coordinator.should_stop():      
                    self.trainer.rollout.wait()                        
                    state_buffer, action_buffer, reward_buffer = [], [], [] 
                action = self.trainer.action(state)
                state_, reward, done, _ = self.env.step(action)
                state_buffer.append(state)
                action_buffer.append(action)
                reward_buffer.append(reward)                  
                state = state_
                episode_reward += reward
                episode_steps += 1

                self.trainer.update_counter += 1              
                if t == self.trainer.episode_length - 1 or self.trainer.update_counter >= self.trainer.batch_size or done:

                    if self.trainer.current_episode >= self.trainer.max_episodes:
                        self.trainer.tf_coordinator.request_stop()
                        break

                    value = self.trainer.get_critic_output(state_)
                    self.trainer.total_steps += episode_steps
                    # Discounted reward
                    discounted_r = []
                    for reward in reward_buffer[::-1]:
                        value = reward + self.trainer.gamma * value
                        discounted_r.append(value)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(state_buffer), np.vstack(action_buffer), np.array(discounted_r)[:, np.newaxis]
                    state_buffer, action_buffer, reward_buffer = [], [], []
                    self.trainer.queue.put(np.hstack((bs, ba, br)))         
                    if self.trainer.update_counter >= self.trainer.batch_size:
                        self.trainer.rollout.clear()       
                        self.trainer.update_event.set()          
                    
                    break
          
            if len(self.trainer.running_reward) == 0: 
                self.trainer.running_reward.append(episode_reward)
            else: 
                self.trainer.running_reward.append(self.trainer.running_reward[-1]*0.9+episode_reward*0.1)
            self.trainer.episode_reward.append(episode_reward)
            self.trainer.current_episode += 1

            if self.trainer.verbose:
                self.status(episode_steps)
                

    def status(self, episode_steps):
        print('{0:.1f}%'.format(float(self.trainer.current_episode)/float(self.trainer.max_episodes)*100),
                self.trainer.current_episode,
            ' | Worker %i' % self.worker_name,
            ' | Ep reward: %.2f' % self.trainer.episode_reward[-1],
            ' | Discounted reward: %.2f' % self.trainer.running_reward[-1],
            ' | S: {}'. format(self.trainer.total_steps),
            ' | S/s: {}'.format(self.trainer.total_steps/(time.time() - self.trainer.time)),
            ' | Ep Steps: {}'.format(episode_steps) )
