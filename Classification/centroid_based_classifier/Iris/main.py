import numpy as np
import pandas as pd

read_iris_train_set = pd.read_csv("iris_train.csv", header=None)
read_iris_test_set = pd.read_csv("iris_test.csv", header=None);

# data info
read_iris_train_set.info()
read_iris_test_set.info()

print(read_iris_test_set)
print(read_iris_train_set)

# convert 'DataFrame' to 'numpy'
dataFrame_irs_train_set = read_iris_train_set[:]
dataFrame_irs_test_set = read_iris_test_set[:]
iris_train_set = np.array(dataFrame_irs_train_set, dtype = 'float32');
iris_test_set = np.array(dataFrame_irs_test_set, dtype = 'float32');

print(iris_train_set.shape)
print(iris_test_set.shape)

x1, y1, z1, count1 = 0., 0., 0., 0
x2, y2, z2, count2 = 0., 0., 0., 0
x3, y3, z3, count3 = 0., 0., 0., 0

def meanVector(x, y, z, count):
    result = np.array([x, y, z])
    result = result/count
    return result

for i in range(120):
    label = iris_train_set[i, 4]
    if label == 1:
        count1 = count1+1
        x1 = x1 + iris_train_set[i, 0]
        y1 = y1 + iris_train_set[i, 1]
        z1 = z1 + iris_train_set[i, 2]
    elif label == 2:
        x2 = x2 + iris_train_set[i, 0]
        y2 = y2 + iris_train_set[i, 1]
        z2 = z2 + iris_train_set[i, 2]
        count2 = count2+1
    else:
        x3 = x3 + iris_train_set[i, 0]
        y3 = y3 + iris_train_set[i, 1]
        z3 = z3 + iris_train_set[i, 2]
        count3 = count3+1

meanVector(x1, y1, z1, count1)
meanVector(x2, y2, z2, count2)
meanVector(x3, y3, z3, count3)





