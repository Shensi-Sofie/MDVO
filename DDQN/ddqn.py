import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import random
from collections import deque
import time
import numpy as np

STATE_DIM = 4
ACTION_NUM = 9
GAMMA = 0.9  # discount factor for target Q
learning_rate = 0.001

INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 128
START_STEP = 1000  # start training step
EPISODE = 200  # Episode limitation
STEP = 65  # Step limitation in an episode
MODEL_PATH = '../results/ddqn_2000ep'


class DDQN(object):

    def __init__(self, act_num, index):
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = 20
        self.action_dim = act_num
        self.index = index

        self.create_Q_network()
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.his_cost = []

    def create_Q_network(self):
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        with tf.variable_scope(str(self.index)+'current_net'):
            w1 = self.weight_variable([self.state_dim, 64])
            b1 = self.bias_variable([64])
            h_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
            w2 = self.weight_variable([64, self.action_dim])
            b2 = self.bias_variable([self.action_dim])
            self.Q_value = tf.matmul(h_layer, w2) + b2

        with tf.variable_scope(str(self.index)+'target_net'):
            w1t = self.weight_variable([self.state_dim, 64])
            b1t = self.bias_variable([64])
            h_layer_t = tf.nn.relu(tf.matmul(self.state_input, w1t) + b1t)
            w2t = self.weight_variable([64, self.action_dim])
            b2t = self.bias_variable([self.action_dim])
            self.target_Q_value = tf.matmul(h_layer_t, w2t) + b2t

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(self.index)+'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=str(self.index)+'current_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def perceive(self, state, action, reward, next_state):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.pop()

        self.replay_buffer.append((state, one_hot_action, reward, next_state))

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        current_Q_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        max_action_next = np.argmax(current_Q_batch, axis=1)
        target_Q_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})

        for i in range(0, BATCH_SIZE):
            target_Q_value = target_Q_batch[i, max_action_next[i]]
            y_batch.append(reward_batch[i] + GAMMA * target_Q_value)
        # Step3: train
        train_loss, _ = self.session.run([self.cost, self.optimizer], feed_dict={self.y_input: y_batch,
                                                                                 self.action_input: action_batch,
                                                                                 self.state_input: state_batch})
        self.his_cost.append(train_loss)

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [state]})[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={self.state_input: [state]})[0])

    def update_target_q_network(self):
        self.session.run(self.target_replace_op)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def return_cost(self):
        return self.his_cost


