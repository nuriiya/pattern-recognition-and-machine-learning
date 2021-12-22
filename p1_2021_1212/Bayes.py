import random

import numpy as np


def get_ill():
    """
    以均值2000，标准差1000生成患病数据
    :return:
    """
    ans = int(random.normalvariate(2000, 1000))
    label = 1
    if ans <= 0:
        return get_ill()
    return ans, label


def get_not_ill():
    """
    以均值7000，标准差3000生成不患病数据
    :return:
    """
    ans = int(random.normalvariate(7000, 3000))
    label = 0
    if ans <= 0:
        return get_not_ill()
    return ans, label


def getOneData():
    """
    以99.5%概率生成不患病和0.5%患病
    :return:
    """
    rd = random.random()
    if rd < 0.005:
        return get_ill()
    else:
        return get_not_ill()


def normal_probability(x, mean, var):
    '''
    计算正态分布情况下，x的概率
    :param x: 输入值
    :param mean: 正态分布均值
    :param var: 正态分布方差
    :return: 输入为x时的条件概率
    '''
    return np.exp(- ((x - mean) ** 2 / (2 * var))) / np.sqrt(2 * np.pi * var)


def get_mean_and_var(arr):
    """
    获取数组的标准差和方差
    :param arr:
    :return:
    """
    if len(arr) == 0:
        return 0, 0
    arr = np.array(arr)
    return int(np.mean(arr)), int(np.var(arr))


if __name__ == '__main__':
    total_num = 100
    train_num = int(total_num * 0.7)
    test_num = int(total_num * 0.3)

    count = 0
    ill_arr = []
    not_ill_arr = []
    for i in range(train_num):
        if i == 0:  # 确保至少一条患病
            v, l = get_ill()
        else:
            v, l = getOneData()
        if l:
            ill_arr.append(v)
        else:
            not_ill_arr.append(v)
    ill_mean, ill_var = get_mean_and_var(ill_arr)
    not_ill_mean, not_ill_var = get_mean_and_var(not_ill_arr)

    ill_var = not_ill_var

    not_ill_rate = round(len(not_ill_arr) / (len(not_ill_arr) + len(ill_arr)), 4)
    ill_rate = 1 - not_ill_rate
    print("训练结果：")
    print("患病白细胞均值：" + str(ill_mean), "患病白细胞方差：" + str(ill_var))
    print("未患病白细胞均值：" + str(not_ill_mean), "未患病白细胞方差：" + str(not_ill_var))
    print("未患病比率：" + str(not_ill_rate), "患病比率：" + str(ill_rate))
    min_err = 0
    min_loss = 0
    print("测试结果(类别1=患病)：")
    for i in range(test_num):
        if i == 0:
            v, l = get_ill()
        else:
            v, l = getOneData()
        p_not_ill = normal_probability(v, not_ill_mean, not_ill_var)
        p_ill = normal_probability(v, ill_mean, ill_var)
        p_x_0 = p_not_ill * not_ill_rate
        p_x_1 = p_ill * ill_rate
        print("白细胞浓度：" + str(v) + "\t类别：" + str(l) + "\t类别1的概率：" + str(p_x_1) + "\t类别0的概率：" + str(p_x_0), end="")
        if (p_x_0 > p_x_1 and l == 1) or (p_x_1 > p_x_0 and l == 0):
            min_err += 1
            print("\t 最小错误率分类错误", end="")
        if (100 * p_x_0 > p_x_1 and l == 1) or (p_x_1 > p_x_0 * 100 and l == 0):
            min_loss += 1
            print("\t 最小风险分类错误错误", end="")
        print()

    min_err /= test_num
    min_loss /= test_num
    print("贝叶斯最小错误率分类错误率：" + str(min_err) + "\t最小风险分类错误率:" + str(min_loss))
