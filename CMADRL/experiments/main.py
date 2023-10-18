import argparse
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tf_slim as layers
import time
from gym import spaces
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from utility import *
from maddpg_plot import *
import random

n_agent = 16
mec_num = 5
actions_num = mec_num * 10 + 1
BATCH_SIZE = 64
MAX_EP = 2000
EP_LEN = 640
GAMMA = 0.99
TRAIN = True
REPLAY_BUFFER_SIZE = 20000

RESULT_PATH_PREFIX = '../../results/maddpg_2000ep_'
label = ["0", '1', '2', '3', '4', '5', '6', '7', '8']
label1 = ["0", '1', '2', '3', '4', '5', '6', '7', '8', 'mean']
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}


def initial_mean_act():
    mean = []
    for i in range(n_agent):
        x = np.zeros((1, actions_num))
        mean.append(x)
    return mean


class MecAgent(object):
    def __init__(self, index):
        self.mindex = index
        self.require_num = 0

    @property
    def index(self):
        return self.mindex

    def add_require_num(self):
        self.require_num += 1

    def edge_total_delay(self, new_task, data_size):
        try:
            compute_delay = new_task / (e_pc / self.require_num)
        except ZeroDivisionError:
            print("require_num=0, 不需要" + self.index + "号边缘服务器计算")
        x = d_trans_power / WHITE_NOISE * (100 ** 4)
        rate = W_DE * np.log2(1 + x)
        trans_delay = data_size / rate
        delay = compute_delay + trans_delay
        return delay

    def reset(self):
        self.require_num = 0


class VehicleAgent(object):
    def __init__(self, index):
        self.index = index
        self.data_size = 0
        self.task_density = 0
        self.dead_line = 0
        self.local_wait = 0
        self.local_compute = 0
        self.edge_delay = 0
        self.local_delay = 0
        self.neighbour_number = 0
        self.neighbour_index = []

        self.action = np.random.randint(0, actions_num)
        self.mean_action = np.zeros((1, actions_num))
        self.mec_index = 0
        self.offload_ratio = 0
        self.total_len = 0

        self.local = 0
        self.offload = 0

        self.buffer = []
        name = 'agent' + str(self.index)

        self.pg_loss, self.vf_loss, self.ent_loss, self.total_loss = [], [], [], []
        self.ep_reward, self.ep_delay, self.ep_local_waits, self.ep_off_delay = [], [], [], []
        self.ep_local_data = []

    def get_neighbours(self, x1, x2, x3, x4):
        self.neighbour_index = [x1, x2, x3, x4]
        for n in self.neighbour_index:
            if n != -1:
                self.neighbour_number += 1

    def generate_task(self):
        self.data_size = round(np.random.poisson(lam=5))
        self.task_density = random.choice([100, 200, 300, 400])
        self.dead_line = random.choice([0.1, 0.2])

    def get_observation(self, neighbour_tasks):
        self.observ = []
        self.observ.append(self.data_size)
        self.observ.append(self.task_density)
        self.observ.append(self.dead_line)
        self.observ.append(self.local_wait)
        self.observ.extend(neighbour_tasks)
        self.observ = np.array(self.observ, ndmin=2, dtype=float)
        return self.observ

    # 此函数必须在get_action（）之后运行
    # 返回值：选择的mec，卸载的计算总量，卸载的数据量
    def parse_action(self):
        if self.action == 0:
            self.mec_index = 0
            self.offload_ratio = 0
            self.offload = 0
        else:
            self.mec_index = int(self.action // 11) + 1
            if (self.action % 10) == 0:
                self.offload_ratio = 1
            else:
                self.offload_ratio = float((self.action % 10) * 0.1)
            self.offload = self.data_size * self.offload_ratio
            self.local = self.data_size - self.offload
        return self.mec_index, self.offload*self.task_density, self.offload, self.local

    # 此函数必须在parse_action之后运行
    # 返回值：local_wait
    def excute_local_task(self):
        self.local_compute = (self.local * self.task_density) / d_pc
        self.local_delay = self.local_compute + self.local_wait
        self.total_len += self.local * self.task_density
        self.total_len -= STEPDUR * d_pc
        if self.total_len <= 0:
            self.total_len = 0
        self.local_wait = self.total_len / d_pc
        return self.local_wait

    def update_edge_delay(self, new_value):
        self.edge_delay = new_value
        return self.edge_delay

    def return_action(self):
        return self.action

    def save_ep_results(self):
        # ep_r
        ep_r = 0
        for l in self.buffer[-EP_LEN:]:
            ep_r += l[0]
        ep_r = ep_r / EP_LEN
        self.ep_reward.append(ep_r)

        # ep_delay
        ep_d = 0
        for k in self.buffer[-EP_LEN:]:
            ep_d += k[1]
        ep_d = ep_d / EP_LEN
        self.ep_delay.append(ep_d)

        # ep_local_waits
        ep_w = 0
        for m in self.buffer[-EP_LEN:]:
            ep_w += m[2]
        ep_w = ep_w / EP_LEN
        self.ep_local_waits.append(ep_w)

        # ep_edge_delay
        ep_edge = 0
        for n in self.buffer[-EP_LEN:]:
            ep_edge += n[3]
        ep_edge = ep_edge / EP_LEN
        self.ep_off_delay.append(ep_edge)

        # ep_local_data
        ep_l_data = 0
        for o in self.buffer[-EP_LEN:]:
            ep_l_data += o[4]
        ep_l_data = ep_l_data / EP_LEN
        self.ep_local_data.append(ep_l_data)

    def return_ep_results(self):
        self.ep_reward = moving_average(self.ep_reward)
        self.ep_delay = moving_average(self.ep_delay)
        self.ep_local_waits = moving_average(self.ep_local_waits)
        self.ep_off_delay = moving_average(self.ep_off_delay)
        self.ep_local_data = moving_average(self.ep_local_data)
        return self.ep_reward, self.ep_delay, self.ep_local_waits, self.ep_off_delay, self.ep_local_data

    def reward_scaling(self):
        re_buffer = []
        for i in self.buffer[-EP_LEN:]:
            re_buffer.append(i[3])
        re_buffer = np.array(re_buffer, dtype=float)
        std = np.std(re_buffer, ddof=1)
        max = np.max(re_buffer)
        for i in self.buffer[-EP_LEN:]:
            i[3] = np.clip(i[3]/std, -max, max)

    def reset(self):
        self.data_size = 0
        self.task_density = 0
        self.dead_line = 0
        self.local_wait = 0
        self.local_compute = 0
        self.edge_delay = 0
        self.local_delay = 0
        self.mec_index = 0
        self.offload_ratio = 0
        self.total_len = 0
        self.local = 0
        self.offload = 0


class OffloadingRunner(object):
    def __init__(self, agents_num):
        self.agents = []
        self.mec_agents = []
        self.agents_num = agents_num
        self.mec_num = mec_num
        self.n = agents_num
        self.action_space = [spaces.Discrete(actions_num) for i in range(agents_num)]
        for i in range(self.agents_num):
            self.agents.append(VehicleAgent(i))  # veh index从0开始
        for j in range(1, self.mec_num+1):
            self.mec_agents.append(MecAgent(j))  # mec index从1开始

        if self.agents_num == 4:
            self.agents[0].get_neighbours(-1, 2, -1, 1)
            self.agents[1].get_neighbours(-1, 3, 0, -1)
            self.agents[2].get_neighbours(0, -1, -1, 3)
            self.agents[3].get_neighbours(1, -1, 2, -1)
        elif self.agents_num == 6:
            self.agents[0].get_neighbours(-1, 3, -1, 1)
            self.agents[1].get_neighbours(-1, 4, 0, 2)
            self.agents[2].get_neighbours(-1, 5, 1, -1)
            self.agents[3].get_neighbours(0, -1, -1, 4)
            self.agents[4].get_neighbours(1, -1, 3, 5)
            self.agents[5].get_neighbours(2, -1, 4, -1)
        elif self.agents_num == 9:
            self.agents[0].get_neighbours(-1, 3, -1, 1)
            self.agents[1].get_neighbours(-1, 4, 0, 2)
            self.agents[2].get_neighbours(-1, 5, 1, -1)
            self.agents[3].get_neighbours(0, 6, -1, 4)
            self.agents[4].get_neighbours(1, 7, 3, 5)
            self.agents[5].get_neighbours(2, 8, 4, -1)
            self.agents[6].get_neighbours(3, -1, -1, 7)
            self.agents[7].get_neighbours(4, -1, 6, 8)
            self.agents[8].get_neighbours(5, -1, 7, -1)
        elif self.agents_num == 12:
            self.agents[0].get_neighbours(-1, 4, -1, 1)
            self.agents[1].get_neighbours(-1, 5, 0, 2)
            self.agents[2].get_neighbours(-1, 6, 1, 3)
            self.agents[3].get_neighbours(-1, 7, 2, -1)
            self.agents[4].get_neighbours(0, 8, -1, 5)
            self.agents[5].get_neighbours(1, 9, 4, 6)
            self.agents[6].get_neighbours(2, 10, 5, 7)
            self.agents[7].get_neighbours(3, 11, 6, -1)
            self.agents[8].get_neighbours(4, -1, -1, 9)
            self.agents[9].get_neighbours(5, -1, 8, 10)
            self.agents[10].get_neighbours(6, -1, 9, 11)
            self.agents[11].get_neighbours(7, -1, 10, -1)
        elif self.agents_num == 16:
            self.agents[0].get_neighbours(-1, 4, -1, 1)
            self.agents[1].get_neighbours(-1, 5, 0, 2)
            self.agents[2].get_neighbours(-1, 6, 1, 3)
            self.agents[3].get_neighbours(-1, 7, 2, -1)
            self.agents[4].get_neighbours(0, 8, -1, 5)
            self.agents[5].get_neighbours(1, 9, 4, 6)
            self.agents[6].get_neighbours(2, 10, 5, 7)
            self.agents[7].get_neighbours(3, 11, 6, -1)
            self.agents[8].get_neighbours(4, -1, -1, 9)
            self.agents[9].get_neighbours(5, -1, 8, 10)
            self.agents[10].get_neighbours(6, -1, 9, 11)
            self.agents[11].get_neighbours(7, -1, 10, -1)
            self.agents[12].get_neighbours(8, -1, -1, 13)
            self.agents[13].get_neighbours(9, -1, 8, 14)
            self.agents[14].get_neighbours(10, -1, 13, 15)
            self.agents[15].get_neighbours(11, -1, 14, -1)

    def get_global_observations(self):
        observations = []
        for i in self.agents:
            i.generate_task()
        for j in self.agents:
            neighbour_tasks = []
            for k in j.neighbour_index:
                if k != -1:
                    neighbour_tasks.append(self.agents[k].data_size)
                    neighbour_tasks.append(self.agents[k].task_density)
                    neighbour_tasks.append(self.agents[k].local_wait)
                    neighbour_tasks.append(self.agents[k].dead_line)
                else:
                    neighbour_tasks.extend([0, 0, 0, 0])
            observations.append(j.get_observation(neighbour_tasks))
        return observations

    def step(self, acts):
        rewards, delays = [], []
        mec_index = []
        offload_task = []
        offload_data_size = []
        local_wait = []
        edge_delay = []
        local_data = []
        for i, j in zip(self.agents, range(len(self.agents))):
            ac = acts[j]
            mec, off, data, ldata = i.parse_action()
            local_w = i.excute_local_task()
            mec_index.append(mec)
            offload_task.append(off)
            offload_data_size.append(data)
            local_data.append(ldata)
            local_wait.append(local_w)

        for k in range(len(mec_index)):
            if mec_index[k] != 0:
                self.mec_agents[mec_index[k]-1].add_require_num()
        for l, m in zip(range(len(self.agents)), self.agents):
            if m.mec_index == 0:
                edge_de = 0
            else:
                edge_de = self.mec_agents[m.mec_index-1].edge_total_delay(offload_task[l],
                                                      offload_data_size[l])
            m.update_edge_delay(edge_de)
            edge_delay.append(edge_de)
        for n in self.agents:
            new_delay = max(n.local_delay, n.edge_delay)
            delays.append(new_delay)
            rew = n.dead_line - new_delay
            rewards.append(rew)
        return rewards, delays, local_wait, edge_delay, local_data

    def store_trajectories(self, rewards, delays, local_waits, edge_delays, local_datas):
        for k, i in zip(self.agents, range(len(self.agents))):
            if len(k.buffer) >= REPLAY_BUFFER_SIZE:
                k.buffer.pop(0)
            k.buffer.append([rewards[i], delays[i], local_waits[i], edge_delays[i], local_datas[i]])

    def reset(self):
        for i in self.agents:
            i.reset()
        for j in self.mec_agents:
            j.reset()


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    #parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=640, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default='../models', help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = OffloadingRunner(n_agent)
        env.get_global_observations()
        # Create agent trainers
        obs_shape_n = [(20,) for i in range(n_agent)]
        num_adversaries = 0
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        # Initialize
        U.initialize()
        env.reset()
        t0 = time.time()

        print('Starting iterations...')
        for ep in range(MAX_EP):
            tf.set_random_seed(1)
            obs_n = env.get_global_observations()
            for t in range(EP_LEN):
                # get action
                action_n = [trainers[i].action(obs_n[i][0]) for i in range(n_agent)]
                new_action_n = []
                for i in action_n:
                    new_action_n.append(np.argmax(i))
                # environment step
                rewards, delays, local_wait, edge_delay, local_data = env.step(new_action_n)
                new_obs_n = env.get_global_observations()
                env.store_trajectories(rewards, delays, local_wait, edge_delay, local_data)

                if t == EP_LEN + 1:
                    done_n = [True for i in range(n_agent)]
                else:
                    done_n = [False for i in range(n_agent)]
                terminal = all(done_n)

                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i][0], action_n[i], rewards[i], new_obs_n[i][0], done_n[i], terminal)
                obs_n = new_obs_n

                # 每一步都训练一次
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, t)  # t%100=0时才训练
            for w in env.agents:
                w.save_ep_results()
            env.reset()
            print("**********Episode ", ep, " Finished***********")

        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        df4 = pd.DataFrame()
        df5 = pd.DataFrame()

        for n, j in zip(env.agents, range(len(env.agents))):
            df1[j], df2[j], df3[j], df4[j], df5[j] = n.return_ep_results()

        df1['mean'] = df1.iloc[:, 0:n_agent].mean(axis=1)
        df2['mean'] = df2.iloc[:, 0:n_agent].mean(axis=1)
        df3['mean'] = df3.iloc[:, 0:n_agent].mean(axis=1)
        df4['mean'] = df4.iloc[:, 0:n_agent].mean(axis=1)
        df5['mean'] = df5.iloc[:, 0:n_agent].mean(axis=1)

        df1.to_csv(RESULT_PATH_PREFIX + 'reward.csv')
        df2.to_csv(RESULT_PATH_PREFIX + 'delay.csv')
        df3.to_csv(RESULT_PATH_PREFIX + 'local_wait.csv')
        df4.to_csv(RESULT_PATH_PREFIX + 'edge_delay.csv')
        df5.to_csv(RESULT_PATH_PREFIX + 'local_data.csv')

        running_time = time.time() - t0
        print("\nRunning time: ", running_time)
        plot(RESULT_PATH_PREFIX)


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)

