import time
from MFAC_2 import *
from utility import *
from plot import *
import random

n_agent = 16  # vehicle number
mec_num = 5
actions_num = mec_num * 10 + 1
MAX_EP = 2000  # training episodes
EP_LEN = 640  # time steps in each episode
GAMMA = 0.99  # discount factor
TRAIN = True
REPLAY_BUFFER_SIZE = 20000
RESULT_PATH_PREFIX = '../results/mdvo_2000ep_'
MODEL_PATH = '../models/'
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
        self.mfagent = MFAC(self.buffer, name, self.index, actions_num)
        self.pg_loss, self.vf_loss, self.ent_loss, self.total_loss = [], [], [], []
        self.ep_reward, self.ep_delay, self.ep_local_waits, self.ep_off_delay = [], [], [], []
        self.ep_local_data = []

    def get_neighbours(self, x1, x2, x3, x4):
        self.neighbour_index = [x1, x2, x3, x4]
        for n in self.neighbour_index:
            if n != -1:
                self.neighbour_number += 1

    def generate_task(self):
        self.data_size = round(np.random.poisson(lam=1))
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

    def get_action(self, obs):
        self.action = self.mfagent.choose_action(obs)
        return int(self.action)

    # 此函数必须在get_action（）之后运行
    # 返回值：选择的mec，卸载的计算总量，卸载的数据量
    def parse_action(self):
        if self.action == 0:
            self.mec_index = 0
            self.offload_ratio = 0
            self.offload = 0
            self.local = self.data_size
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

    def return_one_hot_action(self):
        one_hot_action = np.zeros(actions_num)
        one_hot_action[self.action] = 1
        return one_hot_action

    def update_mean_action(self, mean):
        self.mean_action = mean

    def return_action(self):
        return self.action

    def save_ep_results(self):
        ep_r = 0
        for l in self.buffer[-EP_LEN:]:
            ep_r += l[3]
        ep_r = ep_r / EP_LEN
        self.ep_reward.append(ep_r)

        ep_d = 0
        for k in self.buffer[-EP_LEN:]:
            ep_d += k[4]
        ep_d = ep_d / EP_LEN
        self.ep_delay.append(ep_d)

        ep_w = 0
        for m in self.buffer[-EP_LEN:]:
            ep_w += m[5]
        ep_w = ep_w / EP_LEN
        self.ep_local_waits.append(ep_w)

        ep_edge = 0
        for n in self.buffer[-EP_LEN:]:
            ep_edge += n[6]
        ep_edge = ep_edge / EP_LEN
        self.ep_off_delay.append(ep_edge)

        ep_l_data = 0
        for o in self.buffer[-EP_LEN:]:
            ep_l_data += o[7]
        ep_l_data = ep_l_data / EP_LEN
        self.ep_local_data.append(ep_l_data)

    def return_ep_results(self):
        self.ep_reward = moving_average(self.ep_reward)
        self.ep_delay = moving_average(self.ep_delay)
        self.ep_local_waits = moving_average(self.ep_local_waits)
        self.ep_off_delay = moving_average(self.ep_off_delay)
        self.ep_local_data = moving_average(self.ep_local_data)
        return self.ep_reward, self.ep_delay, self.ep_local_waits, self.ep_off_delay, self.ep_local_data

    def return_loss(self):
        return self.pg_loss, self.vf_loss, self.ent_loss, self.total_loss

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
        self.agents = []  # vehicles
        self.mec_agents = []  # mec server
        self.agents_num = agents_num
        self.mec_num = mec_num
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

    def step(self, obs):
        actions, rewards, delays = [], [], []
        mec_index = []
        offload_task = []
        offload_data_size = []
        local_wait = []
        edge_delay = []
        local_data = []
        for i, j in zip(self.agents, range(len(self.agents))):
            ac = i.get_action(obs[j])
            mec, off, data, ldata = i.parse_action()
            local_w = i.excute_local_task()
            actions.append(ac)
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
        return actions, rewards, delays, local_wait, edge_delay, local_data

    def get_mean_actions(self):
        mean_actions = []
        for k in self.agents:
            temp = np.zeros(actions_num)
            counter = 0
            for l in k.neighbour_index:
                if l != -1:
                    counter += 1
                    temp += self.agents[l].return_one_hot_action()
            temp = temp/counter
            temp = np.array(temp, ndmin=2)
            k.update_mean_action(temp)
            mean_actions.append(temp)
        return mean_actions

    def store_trajectories(self, obs, mean_actions, actions, rewards,
                           delays, local_waits, edge_delays, local_datas):
        for k, i in zip(self.agents, range(len(self.agents))):
            if len(k.buffer) >= REPLAY_BUFFER_SIZE:
                k.buffer.pop(0)
            k.buffer.append([obs[i], mean_actions[i], actions[i], rewards[i],
                                 delays[i], local_waits[i], edge_delays[i], local_datas[i]])

    def save_models(self):
        for h in self.agents:
            h.mfagent.save(MODEL_PATH)

    def load_models(self):
        for q in self.agents:
            q.mfagent.load(MODEL_PATH)


if __name__ == '__main__':
    t0 = time.time()
    # 初始化agents，构造拓扑关系
    runner = OffloadingRunner(n_agent)
    # runner.load_models()
    for ep in range(MAX_EP):
        tf.set_random_seed(1)
        # 新回合开始，获取初始化observation
        obs = runner.get_global_observations()
        counter = 0
        for t in range(EP_LEN):  # 641个时间步，每个时间步是0.1s
            # 选择动作、执行动作、计算奖励值
            actions, rewards, delays, local_waits, edge_delays, local_datas = runner.step(obs)
            # 计算每个agent的mean action
            mean_actions = runner.get_mean_actions()
            # get_new_observation
            new_obs = runner.get_global_observations()
            # 保存轨迹数据
            runner.store_trajectories(obs, mean_actions, actions, rewards,
                                      delays, local_waits, edge_delays, local_datas)
            # 更新observation
            obs = new_obs
            counter += 1
            # 每运行batch_size步, 训练一次
            if ep > 5 and counter >= BATCH_SIZE:
                if counter % BATCH_SIZE == 0:
                    for m in runner.agents:
                        p, v, e, to = m.mfagent.train()
                        m.pg_loss.append(p)
                        m.vf_loss.append(v)
                        m.ent_loss.append(e)
                        m.total_loss.append(to)
        # 保存各项评价指标
        for w in runner.agents:
            w.save_ep_results()
        # 排队长度、评价指标归零
        for w in runner.agents:
            w.reset()
        for x in runner.mec_agents:
            x.reset()
        print("**********Episode ", ep, " Finished***********")

    # 保存结果
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    df_pg_loss = pd.DataFrame()
    df_vf_loss = pd.DataFrame()
    df_ent_loss = pd.DataFrame()
    df_total_loss = pd.DataFrame()

    for n, j in zip(runner.agents, range(len(runner.agents))):
        df1[j], df2[j], df3[j], df4[j], df5[j] = n.return_ep_results()
        df_pg_loss[j], df_vf_loss[j], df_ent_loss[j], df_total_loss[j] = n.return_loss()

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

    df_pg_loss.to_csv(RESULT_PATH_PREFIX + 'pg_loss.csv')
    df_vf_loss.to_csv(RESULT_PATH_PREFIX + 'vf_loss.csv')
    df_ent_loss.to_csv(RESULT_PATH_PREFIX + 'ent_loss.csv')
    df_total_loss.to_csv(RESULT_PATH_PREFIX + 'total_loss.csv')

    plot(RESULT_PATH_PREFIX)
    running_time = time.time() - t0
    print("\nRunning time: ", running_time)
