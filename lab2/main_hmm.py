import scipy.io as scio
import numpy as np
from hmmlearn.hmm import GaussianHMM

# mat 格式下的数据类似json格式
train_data_set = scio.loadmat("ParkingTrain.mat")['Train'][0]
test_data_set = scio.loadmat("ParkingTest.mat")['Test'][0]


def preprocess(ds):
    for s in ds:
        s[0] = s[0].astype(int)
    for c in range(2):
        for i in range(len(s[0][c]) - 1, 0, -1):
            s[0][c][i] -= s[0][c][i - 1]
        s[0][c][0] = 0
    return ds


def preprocess_standard(dt, ds):
    allX = []
    allY = []
    for s in dt:
        s[0] = s[0].astype(int)
        allX = np.append(allX, s[0][0])
        allY = np.append(allY, s[0][1])
        data_mean = [np.mean(allX), np.mean(allY)]
        data_std = [np.std(allX), np.std(allY)]
    for s in dt:
        s[0] = s[0].astype(float)
        for c in range(2):
            s[0][c] = (s[0][c] - data_mean[c]) / data_std[c]

    for s in ds:
        s[0] = s[0].astype(float)
        for c in range(2):
            s[0][c] = (s[0][c] - data_mean[c]) / data_std[c]
    return dt, ds


train_data_set, test_data_set = preprocess(train_data_set), preprocess(test_data_set)
train_data_set, test_data_set = preprocess_standard(train_data_set, test_data_set)


NumTrainItems = 5
for NumHiddenStates in range(1, 10):
    cnt = 0
    h = [GaussianHMM(n_components=NumHiddenStates, n_iter=100, tol=0.001) for x in range(6)]
    for id in range(6):
        assert all(train_data_set[x][1][0] == id+1 for x in range(NumTrainItems*id, NumTrainItems*id+NumTrainItems))
        lengths = [len(train_data_set[x][0][0]) for x in range(NumTrainItems*id, NumTrainItems*id+NumTrainItems)]
        observations_0 = np.concatenate([train_data_set[x][0][0] for x in range(NumTrainItems*id, NumTrainItems*id+NumTrainItems)])
        observations_1 = np.concatenate([train_data_set[x][0][1] for x in range(NumTrainItems*id, NumTrainItems*id+NumTrainItems)])
        observations = np.concatenate((np.reshape(observations_0, (-1, 1)), np.reshape(observations_1, (-1, 1))))
        print("这里是长度%s， 这里是观察数据%s"%(lengths, observations))
        h[id].fit(observations, lengths)
    for x in test_data_set:
        inp = np. concatenate ( (np. reshape (x[0][0], (-1,1)), np. reshape (x[0][1], (-1,1))),axis=-1)
        label = x[1][0][0]
        scores = []
        for id in range(6):
            try:
                scores.append(h[id].score(inp))
            except:
                scores.append(-1e18)
        print(scores)
        cnt += (np.argmax(scores)+1 == label)
    print("when NHS is %d, the accuracy is %.6f"%(NumHiddenStates,(cnt/len(test_data_set))))