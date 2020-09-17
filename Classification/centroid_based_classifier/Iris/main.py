import numpy as np
import pandas as pd

read_iris_train_set = pd.read_csv("iris_train.csv", header=None)
read_iris_test_set = pd.read_csv("iris_test.csv", header=None)

# data info
# read_iris_train_set.info()
# read_iris_test_set.info()

# convert 'DataFrame' to 'numpy'
dataFrame_irs_train_set = read_iris_train_set[:]
dataFrame_irs_test_set = read_iris_test_set[:]
iris_train_set = np.array(dataFrame_irs_train_set, dtype='float32')
iris_test_set = np.array(dataFrame_irs_test_set, dtype='float32')


count = np.zeros([3,4])
meanVector = np.zeros([3,4])


for i in range(np.size(iris_train_set,0)):

    if iris_train_set[i][4] == 1:
        meanVector[0] = meanVector[0] + iris_train_set[i][0:-1]
        count[0] = count[0]+1
    elif iris_train_set[i][4] == 2:
        meanVector[1] = meanVector[1] + iris_train_set[i][0:-1]
        count[1] = count[1] + 1
    elif iris_train_set[i][4] == 3:
        meanVector[2] = meanVector[2] + iris_train_set[i][0:-1]
        count[2] = count[2] + 1

meanVector = meanVector/count
distance = np.zeros([3, 1])

def cal_distance_with_meanVector(data,group_num):
    result = np.sqrt(np.sum(np.square(meanVector[group_num] - data[0:-1])))
    return result
correct_count = 0

for i in range(np.size(iris_test_set,0)):
    label = iris_test_set[i][-1]
    distance[0] = cal_distance_with_meanVector(iris_test_set[i], 0)
    distance[1] = cal_distance_with_meanVector(iris_test_set[i], 1)
    distance[2] = cal_distance_with_meanVector(iris_test_set[i], 2)

    if np.argmin(distance)+1 == label:
        correct_count = correct_count+1


print("정확도 : {:0.3f}%".format(correct_count/30*100, "."))
