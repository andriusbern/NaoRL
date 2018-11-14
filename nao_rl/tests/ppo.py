"""
Threaded implementation of PPO
"""

import tensorflow as tf
import numpy as np
import gym, threading, queue
import nao_rl, time


class PPO(object):
    def __init__(self,
                 global_params,
                 actor_layers   = [500,500],
                 critic_layers  = [500],
                 actor_lr       = .00001 ,
                 critic_lr      = .00002,
                 epsilon        = .2,
                 ):

        self.globals = global_params

        # Parameters
        print "Creating dummy environment to obtain the parameters..."
        env = nao_rl.make(self.globals.env_name, 19998)
        self.action_space = env.action_space.shape[0]
        self.state_space = env.observation_space.shape[0]
        self.action_bounds = [env.action_space.low[0], self.action_space.high[0]]
        nao_rl.destroy_instances()
        del env

        self.sess = tf.Session()

        # Input placeholders
        self.tfs = tf.placeholder(tf.float32, [None, self.state_space], 'state')
        self.tfa = tf.placeholder(tf.float32, [None, self.action_space], 'action')

        # Critic network
        l1 = tf.layers.dense(self.tfs, critic_layers[0], tf.nn.relu) 
        for layer in critic_layers[1::]:
            l1 = tf.layers.dense(l1, critic_layers[layer], tf.nn.relu)

        # Value output layer
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(critic_lr).minimize(self.closs)

        #######
        # Actor
        pi, pi_params = self._build_anet('pi', True, actor_layers)
        oldpi, oldpi_params = self._build_anet('oldpi', False, actor_layers)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(actor_lr).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        while not self.globals.tf_coordinator.should_stop():
            if self.globals.current_episode < self.globals.max_episodes:
                self.globals.update_event.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [self.globals.queue.get() for _ in range(self.globals.queue.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :self.state_space], data[:, self.state_space: self.state_space + self.action_space], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # Update actor and critic networks
                for _ in range(self.globals.epochs):
                    self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) 
                    self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})

                self.globals.update_event.clear()        # updating finished
                self.globals.update_counter = 0         # reset counter
                self.globals.rolling_event.set()        # set roll-out available

    def _build_anet(self, name, trainable, layers):
        with tf.variable_scope(name):

            # Hidden layers
            l1 = tf.layers.dense(self.tfs, layers[0], tf.nn.relu, trainable=trainable)
            for layer in layers[1::]:
                l1 = tf.layers.dense(l1, layer, tf.nn.relu, trainable=trainable)

            # Output layer
            mu = tf.layers.dense(l1, self.action_space, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.state_space, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, self.action_bounds[0], self.action_bounds[1])

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, env, global_ppo, global_params, wid):
        self.wid = wid
        self.env = env
        self.ppo = global_ppo
        self.globals = global_params

    def work(self):
        while not self.globals.tf_coordinator.should_stop():
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            state_buffer, action_buffer, reward_buffer = [], [], []
            for t in range(self.globals.episode_length):
                if not self.globals.rolling_event.is_set():                  # while global PPO is updating
                    self.globals.rolling_event.wait()                        # wait until PPO is updated
                    state_buffer, action_buffer, reward_buffer = [], [], []   # clear history buffer, use new policy to collect data
                action = self.ppo.choose_action(state)
                state_, reward, done, _ = self.env.step(action)
                state_buffer.append(state)
                action_buffer.append(action)
                reward_buffer.append(reward)                  
                state = state_
                episode_reward += reward
                episode_steps += 1

                self.globals.update_counter += 1                # count to minimum batch size, no need to wait other workers
                if t == self.globals.episode_length - 1 or self.globals.update_counter >= self.globals.batch_size or done:
                    value = self.ppo.get_v(state_)

                    # Discounted reward
                    discounted_r = []
                    for reward in reward_buffer[::-1]:
                        value = reward + self.globals.gamma * value
                        discounted_r.append(value)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(state_buffer), np.vstack(action_buffer), np.array(discounted_r)[:, np.newaxis]
                    state_buffer, action_buffer, reward_buffer = [], [], []
                    self.globals.queue.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if self.globals.update_counter >= self.globals.batch_size:
                        self.globals.rolling_event.clear()       # stop collecting data
                        self.globals.update_event.set()          # globalPPO update


                    if self.globals.global_episode >= self.globals.max_episodes:
                        self.globals.tf_coordinator.request_stop()
                        break
                    
                    if done: 
                        self.globals.total_steps += episode_steps
                        break

            # record reward changes, plot later
            if len(self.globals.running_reward) == 0: self.globals.running_reward.append(episode_reward)
            else: self.globals.running_reward.append(self.globals.running_reward[-1]*0.9+episode_reward*0.1)
            self.globals.current_episode += 1

            if self.globals.verbose:
                print('{0:.1f}%'.format(float(self.globals.current_episode)/float(self.globals.max_episodes)*100),
                      self.globals.current_episode,
                    ' | W%i' % self.wid,
                    ' | Ep_r: %.2f' % episode_reward,
                    ' | R_r: %.2f' % self.globals.running_reward[-1],
                    ' | S: {}'. format(self.globals.total_steps),
                    ' | S/s: {}'.format(self.globals.total_steps/(time.time() - self.globals.time)))


class GlobalParams:
    def __init__(self,
                 env_name,
                 gamma,
                 episode_length,
                 max_episodes,
                 batch_size,
                 epochs
                 ):

        # Threading and events
        self.update_event, self.rolling_event = threading.Event(), threading.Event()
        self.tf_coordinator = tf.train.Coordinator()
        self.queue = queue.Queue()

        # Numeral
        self.verbose = False
        self.env_name = env_name
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.total_steps = 0
        self.update_counter = 0
        self.current_episode = 0
        self.running_reward = []
        self.time = time.time()
        


class Trainer(object):
    def __init__(self,
                 env_name,
                 n_workers,
                 batch_size,
                 epochs,
                 max_episodes,
                 max_episode_length,
                 actor_layers,
                 critic_layers,
                 actor_lr,
                 critic_lr,
                 epsilon,
                 gamma,
                 ):

        self.globals = GlobalParams(env_name,
                                    gamma,
                                    max_episode_length,
                                    max_episodes,
                                    batch_size,
                                    epochs)

        # Global network
        self.global_ppo = PPO(self.globals,
                              actor_layers,
                              critic_layers,
                              actor_lr,
                              critic_lr,
                              epsilon)

        # Global parameters to be used by workers

        # Workers
        self.workers = []
        for i in range(n_workers):
            env = nao_rl.make(env_name, 19998-i, headless=True)
            worker = Worker(env, self.global_ppo, self.globals, wid=i)
            worker.env.agent.connect(worker.env, worker.env.active_joints) ### IMPROVE
            self.workers.append(worker)


    def run(self):
        try:
            self.globals.update_event.clear()
            self.globals.rolling_event.set()
            threads = []
            for worker in self.workers:          # worker threads
                t = threading.Thread(target=worker.work, args=())
                t.start()                   # training
                threads.append(t)
            # add a PPO updating thread
            threads.append(threading.Thread(target=self.global_ppo.update,))
            threads[-1].start()
            self.globals.tf_coordinator.join(threads)
        except KeyboardInterrupt:
            print "Interrupted!"

    def save(self):
        pass

