from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import struct
import os
import pandas as pd
import numpy as np
import time
import csv

# line 13 ~ 34 :: ubyte to csv
if not os.path.isfile("./MNIST.csv"):

    train_file = open('train-images.idx3-ubyte', 'rb')
    label_file = open('train-labels.idx1-ubyte', 'rb')
    csv_file = open("MNIST.csv", 'w', encoding="utf-8")

    train_file_size = os.path.getsize("train-labels.idx1-ubyte")  # label 수와 x 데이터 수가 같기에 한번만 size check
    writer = csv.writer(csv_file)
    writer.writerow(np.zeros(785).tolist())  # pandas를 위한 더미 행 추가
    x_cursor, y_cursor = train_file.read(16), label_file.read(8)  # cusor 옮겨놓기
    for i in range((train_file_size-8)):
        x_cursor = train_file.read(784)
        y_cursor = label_file.read(1)

        temp_x = list(struct.unpack(len(x_cursor) * 'B', x_cursor))  # byte->int, list형으로 저장
        temp_y = list(struct.unpack(len(y_cursor) * 'B', y_cursor))  # byte->int, list형으로 저장
        temp_y.extend(temp_x)  # 합치기
        writer.writerow(temp_y)  #csv에 작성

    csv_file.close()
    label_file.close()
    train_file.close()

# read data
# pandas로 읽은 후 numpy로 변환하는 것이 numpy로 바로 읽는 것보다 7초 이상 빨랐습니다.
read_MNIST = pd.read_csv("MNIST.csv")  # pandas로 읽음
# read_MNIST.info()
# print(read_MNIST)
np_MNIST = read_MNIST.to_numpy()  # numpy로 변환

# slice data
X = np_MNIST[:, 1:]
Y = np_MNIST[:, 0]
# print(np.shape(X))  # -> 60000

# scaling
X_255 = X/255
X_MinMax = MinMaxScaler().fit(X).transform(X)

# parmeter setting
debug_Mode = False
cv_ = 5

# model setting
linear_model = LinearSVC(max_iter=10000, verbose=debug_Mode)
non_linear_model = SVC(max_iter=10000, verbose=debug_Mode)

# Run
print("-----Cross validate :: cv is {} -----".format(cv_))

start1 = time.perf_counter()
linear_score = cross_val_score(linear_model, X_255, Y, cv=cv_)
end1 = time.perf_counter()
print("linear :: Acurracy is {}, {:.2f}sec".format(np.round(linear_score.mean(), 3), end1-start1))

start2 = time.perf_counter()
non_linear_score = cross_val_score(non_linear_model, X_255, Y, cv=cv_)
end2 = time.perf_counter()
print("non_linear :: Acurracy is {}, {:.2f}sec\n".format(np.round(non_linear_score.mean(), 3), end2-start2))

print("-----Cross validate with Min_Max_scaling:: cv is {} -----".format(cv_))

start3 = time.perf_counter()
linear_score_with_min_max = cross_val_score(linear_model, X_MinMax, Y, cv=cv_)
end3 = time.perf_counter()
print("linear :: Acurracy is {}, {:.2f}sec".format(np.round(linear_score_with_min_max.mean(), 3), end3-start3))

start4 = time.perf_counter()
non_linear_score_with_min_max = cross_val_score(non_linear_model, X_MinMax, Y, cv=cv_)
end4 = time.perf_counter()
print("non_linear :: Acurracy is {}, {:.2f}sec".format(np.round(non_linear_score_with_min_max.mean(), 3), end4-start4))
