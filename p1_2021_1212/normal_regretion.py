# 假设患病白细胞浓度服从均值为2000，
# 标准差为1000的正态分布，未患病白细胞浓度服从均值为7000，
# 标准差 为3000的正态分布，
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
