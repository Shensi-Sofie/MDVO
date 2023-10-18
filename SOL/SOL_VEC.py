import time
from plot import *
from utils import *
from agent import DDQN

n_agent = 9  # vehicle number
mec_num = 4
actions_num = mec_num + 1
STATE_DIM = n_agent + mec_num
MAX_EP = 2000  # training episodes
EP_LEN = 640  # time steps in each episode
BATCH_SIZE = 64
GAMMA = 0.99  # discount factor
TRAIN = True
REPLAY_BUFFER_SIZE = 20000
RESULT_PATH_PREFIX = '../results/sol_2000ep_'
MODEL_PATH = '../models/sol/'
done = 0


class MecAgent(object):
    def __init__(self, index):
        self.mindex = index
        self.require_num = 0
        self.ca = e_pc

    @property
    def index(self):
        return self.mindex

    @property
    def capacity(self):
        return self.ca

    def add_require_num(self):
        self.require_num += 1

    def edge_total_delay(self, new_task, data_size):
        try:
            compute_delay = new_task / (e_pc / self.require_num)
        except ZeroDivisionError:
            print("require_num=0, 不需要" + self.index + "号边缘服务器计算")
        x1 = d_trans_power / WHITE_NOISE * (100 ** 4)
        rate1 = W_DE * np.log2(1 + x1)
        trans_delay_1 = data_size / rate1
        delay = compute_delay + trans_delay_1
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
        self.qos_level = 0

        self.action = np.random.randint(0, actions_num)
        self.mec_index = 0
        self.total_len = 0

        self.local = 0
        self.offload = 0

        self.buffer = []
        name = 'agent' + str(self.index)
        self.ep_reward, self.ep_delay, self.ep_local_waits, self.ep_off_delay = [], [], [], []
        self.ep_local_data = []

    @property
    def QoS_level(self):
        return self.qos_level

    def generate_task(self):
        self.data_size = round(np.random.poisson(lam=5))
        self.task_density = random.choice([100, 200, 300, 400])
        self.dead_line = random.choice([0.1, 0.2])

    def get_action(self, ac):
        self.action = ac

    # 此函数必须在get_action（）之后运行
    # 返回值：选择的mec，卸载的计算总量，卸载的数据量
    def parse_action(self):
        if self.action == 0:
            self.mec_index = 0
            self.offload = 0
            self.local = self.data_size
        else:
            self.mec_index = int(self.action)
            self.offload = self.data_size
            self.local = 0
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

    def update_qos_level(self):
        if self.mec_index == 0:
            self.qos_level = 1-(self.local_delay/self.dead_line)
        else:
            self.qos_level = 1-(self.edge_delay/self.dead_line)

    def save_ep_results(self):
        # ep_r
        ep_r = 0
        for l in self.buffer[-EP_LEN:]:
            ep_r += l[2]
        ep_r = ep_r / EP_LEN
        self.ep_reward.append(ep_r)

        # ep_delay
        ep_d = 0
        for k in self.buffer[-EP_LEN:]:
            ep_d += k[3]
        ep_d = ep_d / EP_LEN
        self.ep_delay.append(ep_d)

        # ep_local_waits
        ep_w = 0
        for m in self.buffer[-EP_LEN:]:
            ep_w += m[4]
        ep_w = ep_w / EP_LEN
        self.ep_local_waits.append(ep_w)

        # ep_edge_delay
        ep_edge = 0
        for n in self.buffer[-EP_LEN:]:
            ep_edge += n[5]
        ep_edge = ep_edge / EP_LEN
        self.ep_off_delay.append(ep_edge)

        # ep_local_data
        ep_l_data = 0
        for o in self.buffer[-EP_LEN:]:
            ep_l_data += o[6]
        ep_l_data = ep_l_data / EP_LEN
        self.ep_local_data.append(ep_l_data)

    def return_ep_results(self):
        self.ep_reward = moving_average(self.ep_reward)
        self.ep_delay = moving_average(self.ep_delay)
        return self.ep_reward, self.ep_delay

    def reset(self):
        self.data_size = 0
        self.task_density = 0
        self.dead_line = 0
        self.local_wait = 0
        self.local_compute = 0
        self.edge_delay = 0
        self.local_delay = 0
        self.mec_index = 0
        self.total_len = 0
        self.local = 0
        self.offload = 0


class OffloadingRunner(object):
    def __init__(self, vehicles_num, mec_num):
        self.observation = []
        self.actions = []
        self.vehicles = []
        self.mecs = []
        self.vehicles_num = vehicles_num
        self.mec_num = mec_num
        self.controller = DDQN((vehicles_num+mec_num), vehicles_num, (mec_num+1), 0.00001, 'cpu')
        for i in range(self.vehicles_num):
            self.vehicles.append(VehicleAgent(i))  # veh index从0开始
        for j in range(1, self.mec_num+1):
            self.mecs.append(MecAgent(j))  # mec index从1开始

    def get_global_observations(self):
        observations = []
        for i in self.vehicles:
            i.generate_task()
            observations.append(i.QoS_level)
        for j in self.mecs:
            observations.append(j.capacity)
        return observations

    def step(self, obs, eps):
        rewards, delays = [], []
        mec_index = []
        offload_task = []
        offload_data_size = []
        local_wait = []
        edge_delay = []
        local_data = []
        self.actions = self.controller.action(obs, eps)
        for i, j in zip(self.vehicles, range(len(self.vehicles))):
            i.get_action(self.actions[j])
            mec, off, data, ldata = i.parse_action()
            local_w = i.excute_local_task()
            mec_index.append(mec)
            offload_task.append(off)
            offload_data_size.append(data)
            local_data.append(ldata)
            local_wait.append(local_w)

        for k in range(len(mec_index)):
            if mec_index[k] != 0:
                self.mecs[mec_index[k]-1].add_require_num()

        for l, m in zip(range(len(self.vehicles)), self.vehicles):
            if m.mec_index == 0:
                edge_de = 0
            else:
                edge_de = self.mecs[m.mec_index-1].edge_total_delay(offload_task[l],
                                                      offload_data_size[l])
            m.update_edge_delay(edge_de)
            edge_delay.append(edge_de)

        for n in self.vehicles:
            n.update_qos_level()
            new_delay = max(n.local_delay, n.edge_delay)
            delays.append(new_delay)
            rew = n.QoS_level
            rewards.append(rew)
        mean_reward = np.mean(rewards)
        return self.actions, rewards, delays, local_wait, edge_delay, local_data, mean_reward

    def store_trajectories(self, obs, actions, rewards,
                           delays, local_waits, edge_delays, local_datas):
        for k, i in zip(self.vehicles, range(len(self.vehicles))):
            if len(k.buffer) >= REPLAY_BUFFER_SIZE:
                k.buffer.pop(0)
            k.buffer.append([obs[i], actions[i], rewards[i],
                                 delays[i], local_waits[i], edge_delays[i], local_datas[i]])

    def update_model(self, memory):
        self.controller.train_mode(memory, BATCH_SIZE, GAMMA)


if __name__ == '__main__':
    t0 = time.time()
    # 初始化agents，构造拓扑关系
    runner = OffloadingRunner(n_agent, mec_num)
    memory = ReplayBuffer(100000, STATE_DIM, n_agent, 'cpu')
    for ep in range(MAX_EP):
        epsilon = 0.0001
        obs = runner.get_global_observations()
        for t in range(EP_LEN):
            # 选择动作、执行动作、计算奖励值
            actions, rewards, delays, local_waits, edge_delays, local_datas, mean_reward = runner.step(obs, epsilon)
            # get_new_observation
            new_obs = runner.get_global_observations()
            memory.put((obs, actions, mean_reward, new_obs, done))
            if memory.size() > 500:
                runner.update_model(memory)
            # 保存轨迹数据
            runner.store_trajectories(obs, actions, rewards,
                                      delays, local_waits, edge_delays, local_datas)
            obs = new_obs
        for w in runner.vehicles:
            w.save_ep_results()
        # 排队长度、评价指标归零
        for w in runner.vehicles:
            w.reset()
        for x in runner.mecs:
            x.reset()
        print("**********Episode ", ep, " Finished***********")

    # 保存结果
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for n, j in zip(runner.vehicles, range(len(runner.vehicles))):
        df1[j], df2[j] = n.return_ep_results()

    df1['mean'] = df1.iloc[:, 0:n_agent].mean(axis=1)
    df2['mean'] = df2.iloc[:, 0:n_agent].mean(axis=1)

    df1.to_csv(RESULT_PATH_PREFIX + 'reward.csv')
    df2.to_csv(RESULT_PATH_PREFIX + 'delay.csv')

    plot(RESULT_PATH_PREFIX)
    running_time = time.time() - t0
    print("\nRunning time: ", running_time)




