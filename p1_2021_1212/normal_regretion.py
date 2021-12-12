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
risk_ratio = 100

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
            random_sample.append([-1, healthy_sample[c]])
            c += 1
        i += 1
    return random_sample


def verifier(test_label, predict_label):
    count_normal = 0
    count_err_normal = 0
    count_err_sick = 0
    count_ill_real_test_num = 0
    for index, value in enumerate(test_label):
        info = "{:8d} 白细胞浓度{:>15.5f} 实际{:>10} 预测{:>10}". \
            format(index, test_value[index], "患病" if value == 1 else "不患病",
                   "患病" if predict_label[index] == 1 else "不患病")


        if test_label[index] == -1:
            count_normal += 1
        else:
            count_ill_real_test_num += 1
        if test_label[index] != predict_label[index]:
            # 明明患病但是判断为了没患病
            if predict_label[index] == -1:

                count_err_sick += 1
            # 明明没患病但是判断为患病
            else:
                count_err_normal += 1

    info = "总错误率：{}\n".format((count_err_sick + count_err_normal) / len(test_label))
    info += "患病者明明患病但是被判为没患病错误率：{}\n".format(count_err_sick / count_ill_real_test_num)
    info += "患病者明明没有患病但被判为患病错误率：{}".format(count_err_normal / count_ill_real_test_num)

    print(info)

class LeukemiaClassifierModule:
    def __init__(self):
        self.train_mean_normal = 0  # 不患病白细胞浓度均值
        self.train_mean_sick = 0  # 患病白细胞浓度均值
        self.train_std_normal = 0  # 不患病白细胞浓度标准差
        self.train_std_sick = 0  # 患病白细胞浓度标准差
        self.normal_rate = 0  # 不患病概率
        self.sick_rate = 0  # 患病概率

    def Train(self, train_data, train_label):
        normal = []
        sick = []
        normal_num = 0
        for ind, val in enumerate(train_label):
            if -1 == val:
                normal.append(train_data[ind])
                normal_num += 1
            else:
                sick.append(train_data[ind])

        self.train_mean_normal = np.mean(normal)
        self.train_mean_sick = np.mean(sick)
        self.train_std_normal = np.std(normal)
        self.train_std_sick = np.std(sick)
        self.normal_rate = normal_num / len(train_data)
        self.sick_rate = 1 - self.normal_rate

        print("最小错误率最小风险模型训练完成！")
        train_result = "\n训练后不患病白细胞浓度均值：{:.5f};\t训练后不患病白细胞浓度标准差：{:.5f};\n" \
               "训练后" \
                       "患病白细胞浓度均值：{:.5f};\t患病白细胞浓度方差：{:.5f};\n不患病概率：{:.5f};\t患病概率：{:.5f};" \
            .format(self.train_mean_normal, self.train_std_normal,
                    self.train_std_normal, self.train_std_sick, self.normal_rate, self.sick_rate)
        print(train_result)

    def Predict_Minimun_Perror(self, test_data):
        res = []
        for ind, val in enumerate(test_data):
            # p(correct) = p(x|w=2)p(w=2) + p(x|w=1)*p(w=1)
            # 翻译成文字： 就是当病人有白血病时，判断为白血病的几率  加上 正常人没有白血病的几率被判断为没有白血病的几率
            p1 = st.norm(self.train_mean_normal, self.train_std_normal).pdf(val) * self.normal_rate
            p2 = st.norm(self.train_mean_sick, self.train_std_sick).pdf(val) * self.sick_rate
            if p1 < p2:
                res.append(1)
            else:
                res.append(-1)

        return res

    def Predict_Minimun_Risk(self, test_data, risk_ratio):
        res = []
        for ind, val in enumerate(test_data):
            # p(correct) = p(x|w=2)p(w=2) + p(x|w=1)*p(w=1)
            # 翻译成文字： 就是当病人有白血病时，判断为白血病的几率  加上 正常人没有白血病的几率被判断为没有白血病的几率
            p1 = st.norm(self.train_mean_normal, self.train_std_normal).pdf(val) * self.normal_rate
            p2 = st.norm(self.train_mean_sick, self.train_std_sick).pdf(val) * self.sick_rate
            # 和最小错误率不同的是这里乘以了一个倍率
            if p1 < risk_ratio * p2:
                res.append(1)
            else:
                res.append(-1)

        return res
if __name__ == "__main__":
    # 生成测试数据
    all_ill_sample = create_ill_sample(ill_mean,ill_standard_deviation, all_subject_number,ill_percent)
    print("生病人数：\n", np.shape(all_ill_sample))
    all_healthy_sample = create_normal_sample(normal_mean, normal_standard_deviation, all_subject_number, normal_percent)
    print("正常人数：\n", np.shape(all_healthy_sample))

    dataset = mix_sample(all_ill_sample, all_healthy_sample)

    train_set = dataset[:int(all_subject_number * train_percent)]
    print("这里是测试集：", np.shape(train_set))
    test_set = dataset[int(all_subject_number * train_percent):]
    print("这里是训练集：", np.shape(test_set))

    # 为数据集贴上标签
    train_value = [x[1] for x in train_set]
    train_label = [x[0] for x in train_set]
    test_value = [x[1] for x in test_set]
    test_label = [x[0] for x in test_set]

    ill_or_not = LeukemiaClassifierModule()
    # 运算出真正的患病的人的均值和标准差
    ill_or_not.Train(train_value, train_label)

    # 最小错误率情况下的train_label
    predict_label = ill_or_not.Predict_Minimun_Perror(test_value)
    verifier(test_label, predict_label)
    # 最小风险决策下的train——label
    risk_predict_label = ill_or_not.Predict_Minimun_Risk(test_value, 100)
    verifier(test_label, risk_predict_label)

