# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import random
import os

BATCH_SIZE = 256


class MFAC:
    def __init__(self, buffer, name, index, act_num):
        self.sess = tf.Session()
        self.buffer = buffer
        self.name = name
        self.index = index

        self.observ_space = 20
        self.num_actions = act_num
        self.reward_decay = 0.95

        self.learning_rate = 0.1

        self.value_coef = 0.05
        self.ent_coef = 10

        with tf.variable_scope(name):
            self.name_scope = tf.get_variable_scope().name
            self._create_network()

        self.sess.run(tf.global_variables_initializer())

    def _create_network(self):
        # input
        input_observ = tf.placeholder(tf.float32, [None, self.observ_space])  # observation
        input_mean_act = tf.placeholder(tf.float32, [None, self.num_actions])  # mean action
        action = tf.placeholder(tf.int32, [None, ])
        reward = tf.placeholder(tf.float32, [None, ])
        hidden_size = 256

        # fully connected actor network
        w_init = tf.random_normal_initializer(0., 0.01)
        h_observ = tf.layers.dense(input_observ, units=hidden_size, activation=tf.nn.relu, kernel_initializer=w_init,
                             bias_initializer=tf.constant_initializer(0.01))
        dense = tf.layers.dense(h_observ, units=hidden_size, activation=tf.nn.relu, kernel_initializer=w_init,
                             bias_initializer=tf.constant_initializer(0.01))
        policy = tf.layers.dense(dense / 0.1, units=self.num_actions, activation=tf.nn.softmax, kernel_initializer=w_init,
                             bias_initializer=tf.constant_initializer(0.01))
        policy = tf.clip_by_value(policy, 1e-10, 1 - 1e-10)

        self.calc_action = tf.multinomial(tf.log(policy), 1)

        # value network
        emb_prob = tf.layers.dense(input_mean_act, units=64, activation=tf.nn.relu, kernel_initializer=w_init,
                             bias_initializer=tf.constant_initializer(0.01))
        dense_prob = tf.layers.dense(emb_prob, units=32, activation=tf.nn.relu, kernel_initializer=w_init,
                             bias_initializer=tf.constant_initializer(0.01))
        concat_layer = tf.concat([h_observ, dense_prob], axis=1)
        dense = tf.layers.dense(concat_layer, units=hidden_size, activation=tf.nn.relu, kernel_initializer=w_init,
                             bias_initializer=tf.constant_initializer(0.01))
        value = tf.layers.dense(dense, units=1)
        value = tf.reshape(value, (-1,))

        action_mask = tf.one_hot(action, self.num_actions)
        advantage = tf.stop_gradient(reward - value)

        log_policy = tf.log(policy + 1e-6)
        log_prob = tf.reduce_sum(log_policy * action_mask, axis=1)

        vf_loss = self.value_coef * tf.reduce_mean(tf.square(reward - value))
        pg_loss = tf.reduce_mean(advantage * log_prob)
        neg_entropy = self.ent_coef * tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=1))
        total_loss = pg_loss + vf_loss + neg_entropy

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(total_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        self.input_observ = input_observ
        self.input_mean_act = input_mean_act
        self.action = action
        self.reward = reward

        self.policy, self.value = policy, value
        self.pg_loss, self.vf_loss, self.reg_loss = pg_loss, vf_loss, neg_entropy
        self.total_loss = total_loss
        print("Create", self.name)

    def get_value(self, obs, mean_act):
        v = self.sess.run(self.value, feed_dict={
            self.input_observ: obs,
            self.input_mean_act: mean_act
        })[0]
        return v

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)

    def choose_action(self, observ):
        action = self.sess.run(self.calc_action, {
            self.input_observ: observ
        })
        return action.astype(np.int32).reshape((-1,))

    def train(self):
        obs = np.empty((0, 20))
        act = []
        mean_act = np.empty((0, self.num_actions))
        re = []
        mini_batch_sample = random.sample(self.buffer, BATCH_SIZE)

        for k in mini_batch_sample:
            obs = np.concatenate((obs, k[0]), axis=0)
            mean_act = np.concatenate((mean_act, k[1]), axis=0)
            act.append(k[2])
            re.append(k[3])

        s_ = np.array(obs[-1], ndmin=2, dtype=float)
        m_act_ = np.array(mean_act[-1], ndmin=2, dtype=float)
        v_s_ = self.sess.run(self.value, feed_dict={
            self.input_observ: s_,
            self.input_mean_act: m_act_
        })[0]

        # 给reward加上折扣因子
        discounted_r = []
        for r in re[::-1]:
            v_s_ = r + self.reward_decay * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        re = np.array(discounted_r)
        act = np.array(act)

        _, total_loss, pg_loss, vf_loss, ent_loss, state_value = self.sess.run(
            [self.train_op, self.total_loss, self.pg_loss, self.vf_loss, self.reg_loss, self.value], feed_dict={
                self.input_observ: obs,
                self.input_mean_act: mean_act,
                self.action: act,
                self.reward: re,
            })

        return pg_loss, vf_loss, ent_loss, total_loss

    def save(self, dir_path):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "mfac_{}".format(self.index))
        saver.save(self.sess, file_path)
        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "mfac_{}".format(self.index))
        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))

