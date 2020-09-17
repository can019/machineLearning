import numpy as np
import pandas as pd

read_iris_train_set = pd.read_csv("iris_train.csv", header=None)
read_iris_test_set = pd.read_csv("iris_test.csv", header=None);

# data info
read_iris_train_set.info()
read_iris_test_set.info()

# convert 'DataFrame' to 'numpy'
dataFrame_irs_train_set = read_iris_train_set[:]
dataFrame_irs_test_set = read_iris_test_set[:]
iris_train_set = np.array(dataFrame_irs_train_set, dtype = 'float32');
iris_test_set = np.array(dataFrame_irs_test_set, dtype = 'float32');

#slice





