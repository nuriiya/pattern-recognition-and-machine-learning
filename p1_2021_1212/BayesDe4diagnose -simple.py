import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def get_ill(num,lower=0):
    """
    以均值2000，标准差1000生成患病数据

    :return:
    """
    mean_ill = 2000
    std_ill = 1000
    sample_ill =np.array([])
    i=0
    while i <=num:
        sample_an_ill = np.random.normal(mean_ill, std_ill)
        if sample_an_ill>=lower:
            sample_ill=np.append(sample_ill,sample_an_ill)
            i=i+1

    bins = np.linspace(min(sample_ill), max(sample_ill), 20)
    plt.hist(sample_ill, bins, density=True, color='r')
    plt.show()

    return sample_ill

def get_normal(num,lower=0):
    """
            以均值7000，标准差3000生成不患病数据
            :return:
            """
    mean_normal = 7000
    std_normal = 3000
    sample_normal =np.array([])
    i=0
    while i <=num:
        sample_a_normal = np.random.normal(mean_normal, std_normal)
        if sample_a_normal >= lower:
            sample_normal=np.append(sample_normal,sample_a_normal)
            i = i+1
    bins = np.linspace(min(sample_normal), max(sample_normal), 20)
    plt.hist(sample_normal, bins, density=True, color='g')
    plt.show()

    return sample_normal
def MLtrain(data):#请修改训练函数
    mean = 0
    std = 0
    return mean, std

def CDFTrain(data):
    return


if __name__ == '__main__':

    num = 1000
    sample_ill_train=get_ill(np.int64(num*0.005*0.7))
    sample_normal_train=get_normal(np.int64(num*0.995*0.7))
    sample_ill_test = get_ill(np.int64(num * 0.5 * 0.3))
    sample_normal_test = get_normal(np.int64(num * 0.5 * 0.3))
    sample_test = np.append(sample_normal_test,sample_ill_test)
    print(np.size(sample_ill_test))
    sample_label =np.append(np.zeros(np.size(sample_normal_test)),np.ones(np.size(sample_ill_test)))

    #请修改训练过程
    m_ill,v_ill = MLtrain(sample_ill_train)
    m_normal,v_normal = MLtrain(sample_normal_train)
    normal_prior = 0.5
    ill_prior = 0.5

    #请写出后验推理过程
    ill_posterior = np.ones(np.size(sample_test))
    normal_posterior = 1-ill_posterior


    #请修改决策过程和错误率计算过程
    print("最小错误率测试结果：" )
    sample_dec = sample_label
    minError_errorrate = 0

    print(np.size(sample_test))
    for i in range(np.size(sample_test)):

        print("白细胞浓度：" + str(sample_test[i]) + "\t患病吗：" + str(sample_dec[i]) + "\t真实："+ str(sample_label[i]>0) + "\t患病的概率："  \
              + str(ill_posterior[i]) + "\t正常的概率：" + str(normal_posterior[i]), end="\n")

    print("最小风险测试结果：")
    sample_dec = sample_label

    minRisk_errorrate = 0

    for i in range(np.size(sample_test)):

        print("白细胞浓度：" + str(sample_test[i]) + "\t患病吗：" + str(sample_dec[i]) + "\t真实："+ str(sample_label[i]>0) +"\t患病的损失："  \
              + str(normal_posterior[i]) + "\t正常的损失：" + str(ill_posterior[i]*100), end="\n")

    print("训练结果：")
    print("患病白细胞均值：" + str(m_ill), "患病白细胞方差：" + str(v_ill))
    print("未患病白细胞均值：" + str(m_normal), "未患病白细胞方差：" + str(v_normal))
    print("未患病比率：" + str(normal_prior), "患病比率：" + str(ill_prior))
    print("评估结果：")
    print("贝叶斯最小错误率分类错误率：" + str(minError_errorrate) + "\t最小风险分类错误率:" + str(minRisk_errorrate))

    print('ok')






