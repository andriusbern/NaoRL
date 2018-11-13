"""
Asynchronous Advantage Actor Critic (A3C), Reinforcement Learning.
"""

import threading
import tensorflow as tf
import numpy as np
import nao_rl



def a3c_experiment(env_name, 
                   max_episodes,
                   n_workers,
                   ):
    pass

env_name = 'nao-bipedal2'
n_workers = 8
max_episodes = 5000
global_net = 'Global_Net'
global_update_iter = 10
gamma = 0.99
beta = 0.005
lr_actor = 0.00005    # learning rate for actor
lr_critic = 0.0001    # learning rate for critic
reward_history = []
global_episode = 0

# n_states = env.observation_statespace.shape[0]
n_states = 14
n_actions = 12
# n_actions = env.action_statespace.shape[0]
bnd = 0.02
action_bounds = [np.array([-bnd for _ in range(12)]), 
                np.array([ bnd for _ in range(12)])]

# del env
# nao_rl.destroy_instances()


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == global_net:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, n_states], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, n_states], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, n_actions], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    self.test = sigma[0]
                    mu, sigma = mu * action_bounds[1], sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = beta * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1)), *action_bounds)
                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 500, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dense(l_a, 300, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            mu = tf.layers.dense(l_a, n_actions, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, n_actions, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 500, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_c = tf.layers.dense(l_c, 300, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        return mu, sigma, v

    def update_global(self, feed_dict):  # run by a local
        _, _, t = SESS.run([self.update_a_op, self.update_c_op, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})


class Worker(object):
    def __init__(self, name, globalAC, sim_port, head):
        self.env = nao_rl.make(env_name, sim_port, headless=head)
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global reward_history, global_episode
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and global_episode < max_episodes:
            s = self.env.reset()
            ep_r = 0
            ep_s = 0
            while True:
                a = self.AC.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                if r == -100: r = -2
                ep_s += 1
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % global_update_iter == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    test = self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(reward_history) == 0:  # record running episode reward
                        reward_history.append(ep_r)
                    else:
                        reward_history.append(0.95 * reward_history[-1] + 0.05 * ep_r)
                    print(
                        self.name,
                        "Ep:", global_episode,
                        "| RR: %.1f" % reward_history[-1],
                        '| EpR: %.1f' % ep_r,
                       # '| var:', test,
                        '| E S: ', ep_s
                    )
                    global_episode += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()
    nao_rl.destroy_instances()
    # nao_ports = [5000 + i for i in range(n_workers)]
    #nao_rl.start_naoqi(nao_ports)

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(lr_actor, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(lr_critic, name='RMSPropC')
        GLOBAL_AC = ACNet(global_net)  # we only need its params
        workers = []
        sim = 19992
        # Create worker
        # head = [False,False,False,False,False,False,False,False]
        head = [True, True, True, True, True, True, True, True]
        for i in range(n_workers):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC, sim-i, head[i]))
            workers[i].env.agent.connect(workers[i].env, workers[i].env.active_joints)

    try:
        COORD = tf.train.Coordinator()
        SESS.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)
    except KeyboardInterrupt:
        print('Interrupted')

    
    import matplotlib.pyplot as plt
    plt.plot(reward_history)
    plt.xlabel('episode')
    plt.ylabel('global running reward')
    plt.show()

