import numpy as np

W_DD = 1
W_DE = 4
W_EC = 8
d_trans_power = 4
e_trans_power = 10
WHITE_NOISE = 10e-14
e_pc = 10e7
d_pc = 10e4
DATA_PRO_DENSITY = 100
DATA_SIZE = 4
STEPDUR = 0.1


def dd_trans_t():
    x = d_trans_power / WHITE_NOISE * (100 ** 4)
    rate = W_DD * np.log2(1 + x)
    trans_t = DATA_SIZE / rate
    return trans_t


def d_pro_t():
    return DATA_SIZE * DATA_PRO_DENSITY/d_pc


def d_pro_num():
    return int((STEPDUR - dd_trans_t()) / d_pro_t())


def de_trans_t():
    x = d_trans_power/WHITE_NOISE*(100**4)
    rate = W_DE * np.log2(1+x)
    trans_t = DATA_SIZE/rate
    return trans_t


def e_pro_t():
    return DATA_SIZE * DATA_PRO_DENSITY/e_pc


def e_pro_num():
    return int((STEPDUR-de_trans_t()) / e_pro_t())


# E-C
def ec_trans_t():
    x = e_trans_power/WHITE_NOISE*(100**4)
    rate = W_EC * np.log2(1+x)
    trans_t = DATA_SIZE/rate
    return trans_t


def ec_trans_num():
    return int((STEPDUR - de_trans_t()) / ec_trans_t())


def moving_average(raw_list):
    new_list = []
    for i in range(len(raw_list)):
        if i == 0:
            new_list.append(raw_list[i])
        else:
            new_list.append(new_list[i-1]*0.9 + raw_list[i]*0.1)
    return new_list


def test_delay(new_task, data_size):
    compute_delay = new_task/ (e_pc / 2)   # ES的运算能力10e5MHz
    x = d_trans_power / WHITE_NOISE * (100 ** 4)
    rate = W_DE * np.log2(1 + x)
    trans_delay = data_size / rate
    delay = compute_delay + trans_delay
    return delay


