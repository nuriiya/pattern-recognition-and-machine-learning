# 假设患病白细胞浓度服从均值为2000，
# 标准差为1000的正态分布，未患病白细胞浓度服从均值为7000，
# 标准差为3000的正态分布，
# 患病的人数比例为0.5%，
# 问当白细胞浓度为*时，应该做出什么决策？假设患病判为正常的损失是正常判为患病的损失的100倍。
# 根据假设产生10000条模拟数据，按照7：3划分训练和测试集，
# 并实现贝叶斯最小错误率分类和最小风险分类。
import numpy as np
import scipy.stats as st
import matplotlib as plt
import random

all_subject_number = 10000
train_percent = 0.7
test_percent = 0.3
ill_standard_deviation = 1000
normal_standard_deviation = 3000
ill_mean = 2000
normal_mean = 7000
ill_percent = 0.005
normal_percent = 1 - ill_percent

def create_ill_sample(ill_mean, ill_standard_deviation,  all_subject_number, ill_percent):
    sample_ill = np.array([])
    ill_num = ill_percent * all_subject_number
    i = 0
    while i < int(ill_num):
        ill_leverage = np.random.normal(ill_mean,ill_standard_deviation)
        if ill_leverage <= 0:
            i = i

            continue
        else:
           sample_ill = np.append(sample_ill, ill_leverage)
           i = i + 1
    return sample_ill

def create_normal_sample(normal_mean, normal_standard_deviation,  all_subject_number, normal_percent):
    sample_normal = np.array([])
    normal_num = normal_percent * all_subject_number
    i = 0
    while i < int(normal_num):
        normal_leverage = np.random.normal(normal_mean, normal_standard_deviation)
        if normal_leverage <= 0:
            i = i
            continue
        else:
            sample_normal = np.append(sample_normal, normal_leverage)
            i += 1
    return sample_normal

def mix_sample(ill_sample, healthy_sample):
    random_sample = []
    i = 0
    b = 0
    c = 0
    while i < int(all_subject_number):
        rand = random.random()
        if rand <= ill_percent and b < all_subject_number * ill_percent:
            random_sample.append([1, ill_sample[b]])
            b += 1
        elif c < all_subject_number * normal_percent:
            random_sample.append([0, healthy_sample[c]])
            c += 1
        i += 1
    return random_sample

if __name__ == "__main__":
    all_ill_sample = create_ill_sample(ill_mean,ill_standard_deviation, all_subject_number,ill_percent)
    print("生病人数：\n", np.shape(all_ill_sample))
    all_healthy_sample = create_normal_sample(normal_mean, normal_standard_deviation, all_subject_number, normal_percent)
    print("正常人数：\n", np.shape(all_healthy_sample))

    dataset = mix_sample(all_ill_sample, all_healthy_sample)

    train_set = dataset[:int(all_subject_number * train_percent)]
    print("这里是测试集：", np.shape(train_set))
    test_set = dataset[int(all_subject_number * train_percent):]
    print("这里是训练集：", np.shape(test_set))


