import matplotlib.pyplot as plt
import pandas as pd
label = ["0", '1', '2', '3', '4', '5', '6', '7', '8']
label1 = ["0", '1', '2', '3', '4', '5', '6', '7', '8', 'mean']
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
PREFIX = '../results/ddqn_2000ep_'


def plot(prefix):
    df1 = pd.read_csv(prefix + 'delay.csv')
    df2 = pd.read_csv(prefix + 'reward.csv')
    mean_delay = df1.iloc[-1, -1]
    mean_reward = df2.iloc[-1, -1]
    print("mean_delay:", mean_delay)
    print("mean_reward:", mean_reward)

    plt.figure()
    plt.grid()
    plt.plot(df1['0'], color='red', linestyle='-', linewidth=1.5)
    plt.plot(df1['1'], color='blue', linestyle='--', linewidth=1.5)
    plt.plot(df1['2'], color='gold', linestyle=':', marker='x', linewidth=1.5)
    plt.plot(df1['3'], color='lightseagreen', linestyle=':', marker='|', linewidth=1.5)
    plt.plot(df1['4'], color='tomato', linestyle='-', linewidth=1.5)
    plt.plot(df1['5'], color='navy', linestyle='--', linewidth=1.5)
    # plt.plot(df1['6'], color='magenta', linestyle=':', marker='x', linewidth=1.5)
    # plt.plot(df1['7'], color='cyan', linestyle=':', marker='|', linewidth=1.5)
    # plt.plot(df1['8'], color='gray', linestyle=':', marker='|', linewidth=1.5)
    plt.plot(df1['mean'], color='black', linestyle=':', marker='|', linewidth=1.5)
    plt.legend(label1, prop=font1, loc='upper right', ncol=2)
    plt.ylabel('Average delay', font1)
    plt.xlabel('Training Episode', font1)
    plt.tick_params(labelsize=10)
    # plt.show()

    plt.figure()
    plt.grid()
    plt.plot(df2['0'], color='red', linestyle='-', linewidth=1.5)
    plt.plot(df2['1'], color='blue', linestyle='--', linewidth=1.5)
    plt.plot(df2['2'], color='gold', linestyle=':', marker='x', linewidth=1.5)
    plt.plot(df2['3'], color='lightseagreen', linestyle=':', marker='|', linewidth=1.5)
    plt.plot(df1['4'], color='tomato', linestyle='-', linewidth=1.5)
    plt.plot(df1['5'], color='navy', linestyle='--', linewidth=1.5)
    # plt.plot(df1['6'], color='magenta', linestyle=':', marker='x', linewidth=1.5)
    # plt.plot(df1['7'], color='cyan', linestyle=':', marker='|', linewidth=1.5)
    # plt.plot(df1['8'], color='gray', linestyle=':', marker='|', linewidth=1.5)
    plt.plot(df2['mean'], color='black', linestyle=':', marker='|', linewidth=1.5)
    plt.legend(label1, prop=font1, loc='upper right', ncol=2)
    plt.ylabel('Average reward', font1)
    plt.xlabel('Training Episode', font1)
    plt.tick_params(labelsize=10)
    # plt.show()


if __name__ == '__main__':
    plot(PREFIX)
