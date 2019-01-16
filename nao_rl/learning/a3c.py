"""
Asynchronous Advantage Actor Critic (A3C), Reinforcement Learning.

"""

import multiprocessing
import datetime
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import nao_rl

GAME = 'NaoTracking'
RENDER = False

OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = 1
MAX_GLOBAL_EP = 100000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99
ENTROPY_BETA = 0.02
LR_A = 0.000025    # learning rate for actor
LR_C = 0.00005    # learning rate for critic
GLOBAL_RUNNING_R = []

GLOBAL_EP = 0

class A3C(object):
    def __init__(self, env_name, render, n_workers=8, max_episodes=5000, episode_length=500,
                 epochs=10, entropy_beta=.02, gamma=.99,
                 actor_layers=[250,250], critic_layers=[250],
                 actor_lr=.0001, critic_lr=.0001):



try:
    env = nao_rl.make(GAME)
except:
    env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]
del env


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'state_input')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    self.test = sigma[0]
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_action'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1)), *A_BOUND, name='choose_action')
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
            l_a = tf.layers.dense(self.s, 250, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dense(l_a, 250, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 250, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_c = tf.layers.dense(l_c, 250, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
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
    def __init__(self, name, globalAC):

        try:
            self.env = nao_rl.make(GAME, headless=RENDER)
        except:
            self.env = gym.make(GAME)

        self.name = name

        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP, data
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            ep_s = 0
            while True:
                # if self.name == 'W_0' and total_step % 30 == 0:
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if self.name == 'W_0':
                    data[0].append(self.env.agent.joint_position[0])
                    data[1].append(self.env.agent.joint_angular_v[0])

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
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
                ep_s += 1
                if done:
                    # achieve = '| Achieve' if self.env.unwrapped.hull.position[0] >= 88 else '| -------'
                    avg_r = ep_r / float(ep_s)
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        # achieve,
                        # "| Pos: %i" % self.env.unwrapped.hull.position[0],
                        "| RR: %.1f" % GLOBAL_RUNNING_R[-1],
                        '| EpR: %.1f' % ep_r,
                        '| EpS: %i' % ep_s,
                        '| AvgR: %.3f' % avg_r,
                    )
                    GLOBAL_EP += 1

                    if ep_s > 2000:
                        COORD.should_stop()
                    # if avg_r > .25:
                        # date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                        # algorithm = 'a3c'


                        # model_path = '{}/{}_{}_{}_{}.cpkt'.format(nao_rl.settings.TRAINED_MODELS, GAME, avg_r, date, GLOBAL_EP)
                       
                        # SAVER.save(SESS, model_path)
                        # print 'Trained model saved at {}'.format(model_path)
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    import time
    time.sleep(5)
    data = [[],[]]

    try:
        worker_threads = []
        # for worker in workers:
        #     job = lambda: worker.work()
        #     t = threading.Thread(target=job)
        #     t.start()
        #     worker_threads.append(t)
        # COORD.join(worker_threads)

        for worker in workers:
            t = threading.Thread(target=worker.work, args=())
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads, ignore_live_threads=True, stop_grace_period_secs=5)
    except KeyboardInterrupt:
        print 'Stopped'
        COORD.should_stop()


    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    algorithm = 'a3c'

    model_path = '{}/{}_{}_{}.cpkt'.format(nao_rl.settings.TRAINED_MODELS, GAME, algorithm, date)
    saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=1)
    saver.save(SESS, model_path)
    print 'Trained model saved at {}'.format(model_path)


    import matplotlib.pyplot as plt
    # plt.plot(GLOBAL_RUNNING_R)
    # plt.xlabel('episode')
    # plt.ylabel('global running reward')
    # plt.show()
    plt.plot(data[0])
    plt.plot(data[1])
    plt.show()
