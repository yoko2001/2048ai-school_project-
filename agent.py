import tensorflow as tf
import numpy as np 
import random
from collections import deque

np.random.seed(233)
tf.set_random_seed(233)


class DQN(object):
    def __init__(
        self, 
        n_actions, n_feature,
        learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9,
        replace_target_iter = 100, memory_size = 200000, batch_size = 32,
        e_greedy_increment = None, output_graph = False, epsilon_increment = None
    ):  
        self.n_actions = n_actions
        self.n_feature = n_feature
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        #total
        self.learn_step_counter = 0

        #initialize memory space
        self.memory = np.zeros((self.memory_size, n_feature*3))

        #consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.cost_his = []
    def _build_net(self):
        #inputs
        self.s = tf.placeholder(tf.float32, [None, self.n_feature], name= 's') #input state
        self.s_ = tf.placeholder(tf.float32, [None, self.n_feature], name = 's_') #input next state
        self.r = tf.placeholder(tf.float32, [None,], name= 'r') #input reward
        self.a = tf.placeholder(tf.int32, [None,], name='a') #input action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # # ------------------ build evaluate_net ------------------
        # with tf.variable_scope('eval_net'):
        #     e1 = tf.layers.dense(self.s, 16, tf.nn.relu, kernel_initializer=w_initializer,
        #                          bias_initializer=b_initializer, name='e1')
        #     self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
        #                                   bias_initializer=b_initializer, name='q')
        
        # # ------------------ build target_net ------------------
        # with tf.variable_scope('target_net'):
        #     t1 = tf.layers.dense(self.s_, 16, tf.nn.relu, kernel_initializer=w_initializer,
        #                          bias_initializer=b_initializer, name='t1')
        #     self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
        #                                   bias_initializer=b_initializer, name='t2')

        # ------------------ build evaluate_net2 ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 16, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(self.s, 10, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')                     
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 16, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(self.s_, 10, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')                     
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='qr')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis= 1, name="Qmax_s_")  #shape = (None,)
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name = 'TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.stack([s, [float(a), float(r), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,], s_,] )
        #emplace old memory with new memory

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition.reshape(1, 48)
        self.memory_counter += 1
    
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon: 
            #forward feed the observation and get a value for every acitons
            actions_value = self.sess.run(self.q_eval, feed_dict = {self.s: observation})
            action = np.argsort(actions_value)
            return action
        else:
            #sometime we move randomly
            action = np.random.permutation(self.n_actions)
            return action[np.newaxis, :]

    def learn(self):
        #check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')
        
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_feature],
                self.a: batch_memory[:, self.n_feature],
                self.r: batch_memory[:, self.n_feature + 1],
                self.s_: batch_memory[:, -self.n_feature:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
    
    def show_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show() 

if __name__ == '__main__':
    DQN = DQN(4, 16, output_graph=False)