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

all_subject_number = 10000
train_percent = 0.7
test_percent = 0.3
ill_standard_deviation = 1000
normal_standard_deviation = 3000
ill_mean = 2000
normal_mean = 7000
ill_percent = 0.005

def create_ill_sample(normal_mean, normal_standard_deviation,  all_subject_number, ill_percent):
    sample_ill = np.array([])
    ill_num = ill_percent * all_subject_number
    for i in range(0,int(ill_num)):
        ill_leverage = np.random.normal(ill_mean,ill_standard_deviation)
        if(ill_leverage <= 0):
            i -= 1
            continue
        sample_ill = np.append(sample_ill, ill_leverage)
    return sample_ill

def create_normal_sample(ill_mean, ill_standard_deviation,  all_subject_number, ill_percent):
    sample_ill = np.array([])
    ill_num = ill_percent * all_subject_number
    for i in range(0,int(ill_num)):
        ill_leverage = np.random.normal(ill_mean,ill_standard_deviation)
        if(ill_leverage <= 0):
            i -= 1
            continue
        sample_ill = np.append(sample_ill, ill_leverage)
    return sample_ill
if __name__ == "__main__":
    all_ill_sample = create_ill_sample(ill_mean,ill_standard_deviation, all_subject_number,ill_percent)
    print("生病人数：\n", np.shape(all_ill_sample), "\n具体数值：\n", all_ill_sample)
    all_healthy_sample = create_normal_sample()