#!/usr/bin/python3.8

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats

MAX_TRY_TIMES = 100
G_SICK_RATE = 0.005

NORMAL_MEAN = 7000
NORMAL_STD = 3000
SICK_MEAN = 2000
SICK_STD = 1000

BAR_STEP = 500


def Show_Data(train_data, test_data):
    mormal_data = []
    sick_data = []
    for item in train_data:
        if item[0] == 1:
            mormal_data.append(item[1])
        else:
            sick_data.append(item[1])

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.hist(mormal_data, range(0, (int(max(mormal_data) / BAR_STEP) + 1) * BAR_STEP, BAR_STEP))
    plt.figure(1)
    plt.subplot(2, 2, 2)
    plt.hist(sick_data, range(0, (int(max(sick_data) / BAR_STEP) + 1) * BAR_STEP, BAR_STEP))

    mormal_data.clear()
    sick_data.clear()
    for item in test_data:
        if item[0] == 1:
            mormal_data.append(item[1])
        else:
            sick_data.append(item[1])

    plt.figure(1)
    plt.subplot(2, 2, 3)
    plt.hist(mormal_data, range(0, (int(max(mormal_data) / BAR_STEP) + 1) * BAR_STEP, BAR_STEP))
    plt.figure(1)
    plt.subplot(2, 2, 4)
    plt.hist(sick_data, range(0, (int(max(sick_data) / BAR_STEP) + 1) * BAR_STEP, BAR_STEP))
    plt.show()


def Data_Generate(num):
    res = []
    for i in range(num):
        r = random.random()
        if r < G_SICK_RATE:
            # 患病
            for i in range(MAX_TRY_TIMES):
                wbc = random.gauss(SICK_MEAN, SICK_STD)
                if wbc > 0:
                    break
            res.append([-1, wbc])
        else:
            # 不患病
            for i in range(MAX_TRY_TIMES):
                wbc = random.gauss(NORMAL_MEAN, NORMAL_STD)
                if wbc > 0:
                    break
            res.append([1, wbc])
    return res


class BayesClassifierModule:
    def __init__(self):
        self.wbc_con_mean_normal = 0  # 不患病白细胞浓度均值
        self.wbc_con_mean_sick = 0  # 患病白细胞浓度均值
        self.wbc_con_std_normal = 0  # 不患病白细胞浓度标准差
        self.wbc_con_std_sick = 0  # 患病白细胞浓度标准差
        self.normal_rate = 0  # 不患病概率
        self.sick_rate = 0  # 患病概率

    def Train(self, train_data, train_lable):
        normal = []
        sick = []
        normal_count = 0
        for ind, val in enumerate(train_lable):
            if 1 == val:
                normal.append(train_data[ind])
                normal_count += 1
            else:
                sick.append(train_data[ind])

        self.wbc_con_mean_normal = np.mean(normal)
        self.wbc_con_mean_sick = np.mean(sick)
        self.wbc_con_std_normal = np.std(normal)
        self.wbc_con_std_sick = np.std(sick)
        self.normal_rate = normal_count / len(train_data)
        self.sick_rate = 1 - self.normal_rate

        print("贝叶斯分类器模型训练完成！")
        info = "\n不患病白细胞浓度均值：{:.5f};\t不患病白细胞浓度标准差：{:.5f};\n" \
               "患病白细胞浓度均值：{:.5f};\t患病白细胞浓度方差：{:.5f};\n不患病概率：{:.5f};\t患病概率：{:.5f};" \
            .format(self.wbc_con_mean_normal, self.wbc_con_std_normal,
                    self.wbc_con_mean_sick, self.wbc_con_std_sick, self.normal_rate, self.sick_rate)
        print("\033[1;33m", info, "\033[0m")

    def Predict_Minimun_Perror(self, test_data):
        res = []
        for ind, val in enumerate(test_data):
            p1 = stats.norm(self.wbc_con_mean_normal, self.wbc_con_std_normal).pdf(val) * self.normal_rate
            p2 = stats.norm(self.wbc_con_mean_sick, self.wbc_con_std_sick).pdf(val) * self.sick_rate
            if p1 > p2:
                res.append(1)
            else:
                res.append(-1)

        return res

    def Predict_Minimun_Risk(self, test_data, risk_ratio):
        res = []
        for ind, val in enumerate(test_data):
            p1 = stats.norm(self.wbc_con_mean_normal, self.wbc_con_std_normal).pdf(val) * self.normal_rate
            p2 = stats.norm(self.wbc_con_mean_sick, self.wbc_con_std_sick).pdf(val) * self.sick_rate
            if p1 > risk_ratio * p2:
                res.append(1)
            else:
                res.append(-1)

        return res


if __name__ == '__main__':
    count = 10000                                           # 数据集大小
    data_set = Data_Generate(count)                         # 生成数据集
    train_set = data_set[:int(count * 0.7)]                 # 70%作为训练集
    test_set = data_set[int(count * 0.7):]                  # 30%作为测试集

    Show_Data(train_set, test_set)                          # 通过柱状图展示数据分布情况

    mod = BayesClassifierModule()                           # 定义模型
    train_value = [x[1] for x in train_set]
    train_lable = [x[0] for x in train_set]
    test_value = [x[1] for x in test_set]
    test_lable = [x[0] for x in test_set]

    mod.Train(train_value, train_lable)                     # 训练模型

    # 最小错误率决策
    predict_lable = mod.Predict_Minimun_Perror(test_value)

    count_normal = 0
    count_err_normal = 0
    count_err_sick = 0
    for index, value in enumerate(test_lable):
        info = "{:8d} 白细胞浓度：{:>15.5f} 实际：{:>10} 预测：{:>10}". \
            format(index, test_value[index], "患病" if value == -1 else "不患病",
                   "患病" if predict_lable[index] == -1 else "不患病")

        if value == 1:
            count_normal += 1

        if test_lable[index] != predict_lable[index]:
            print("\033[1;43m", info, "\033[0m")
            if value == 1:
                count_err_normal += 1
            else:
                count_err_sick += 1
        else:
            print("\033[1;42m", info, "\033[0m")

    info = "总错误率：{}\n".format((count_err_sick + count_err_normal) / len(test_lable))
    info += "患病者错误率：{}".format(count_err_sick / (len(test_lable) - count_normal))
    print("\033[1;33m", info, "\033[0m")

    # 最小风险决策
    predict_lable = mod.Predict_Minimun_Risk(test_value, 100)

    count_normal = 0
    count_err_normal = 0
    count_err_sick = 0
    for index, value in enumerate(test_lable):
        info = "{:8d} 白细胞浓度{:>15.5f} 实际{:>10} 预测{:>10}". \
            format(index, test_value[index], "患病" if value == -1 else "不患病",
                   "患病" if predict_lable[index] == -1 else "不患病")

        if value == 1:
            count_normal += 1

        if test_lable[index] != predict_lable[index]:
            print("\033[1;43m", info, "\033[0m")
            if value == 1:
                count_err_normal += 1
            else:
                count_err_sick += 1
        else:
            print("\033[1;42m", info, "\033[0m")

    info = "总错误率：{}\n".format((count_err_sick + count_err_normal) / len(test_lable))
    info += "患病者错误率：{}".format(count_err_sick / (len(test_lable) - count_normal))
    print("\033[1;33m", info, "\033[0m")

